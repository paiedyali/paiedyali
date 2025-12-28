# appv3.py — QUADRA + SILAE (OCR auto), PDF synthèse téléchargeable
# Corrections incluses :
# - SILAE : filtre anti-acompte délirant (ex: 2 025 500,00)
# - QUADRA : "Total" puis "Versé employeur" sur ligne suivante (montant sur i / i+1 / i+2 / i+3 / i+4)
# - PDF : 4 carrés lisibles (titre en haut, montant plus bas, wrap automatique)
# - Différence brut/net : calculée sur net reconstitué si acompte trouvé
# - "Organismes sociaux" : total unique (charges salariales + charges patronales + CSG non déductible)

import io
import re
import datetime as dt
import os

import streamlit as st
import pdfplumber
import pytesseract
from pytesseract import Output

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# Import des bibliothèques
import io
import re
import datetime as dt
import os

import streamlit as st
import pdfplumber
import pytesseract
from pytesseract import Output

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas


# ------------------------------------------------------------
# UI
# ------------------------------------------------------------
st.set_page_config(page_title="Lecteur bulletin (Quadra + SILAE)", layout="wide")
st.title("🧾 Ton bulletin de salaire (traduit en français courant)")
st.write("Tu déposes ton bulletin PDF → synthèse simple + export PDF (humour factuel).")
# ------------------------------------------------------------


# ------------------------------------------------------------
# Vérifie si un fichier a été téléchargé
uploaded = st.file_uploader("Dépose ton bulletin de salaire (PDF)", type=["pdf"], key="unique_file_uploader_1")

# Vérification si un fichier a bien été téléchargé avant de l'utiliser
if uploaded is not None:
    # Si un fichier est téléchargé, on continue avec l'analyse
    file_obj = io.BytesIO(uploaded.getvalue())
    st.success("Fichier reçu ✅")

    # Poursuite de la logique pour extraire le texte et analyser le PDF
    status = st.status("Démarrage de l'analyse…", expanded=True)

    status.write("1/6 Lecture du PDF + extraction texte (OCR si besoin)…")
    text, used_ocr, page_images, page_texts, page_ocr_flags = extract_text_auto_per_page(file_obj, dpi=DPI, force_ocr=OCR_FORCE)

    status.write(f"✅ Texte extrait (OCR utilisé: {used_ocr})")
    
    # Poursuite du processus d'analyse
    status.write("2/6 Vérification du document…")
    ok_doc, msg_doc, doc_dbg = validate_uploaded_pdf(page_texts)
    if not ok_doc:
        status.update(label="Analyse interrompue", state="error")
        st.error(msg_doc)
        if DEBUG:
            st.json(doc_dbg)
        st.stop()

    fmt, fmt_dbg = detect_format(text)
    status.write(f"✅ Document valide — format détecté: {fmt}")
    # Suite du traitement...
else:
    # Si aucun fichier n'est téléchargé, on invite l'utilisateur à télécharger un fichier
    st.info("ℹ️ Veuillez télécharger un fichier PDF pour commencer l'analyse.")


st.markdown(
    """
<div style="
    border: 2px solid #1f77b4;
    border-radius: 12px;
    padding: 12px 14px;
    background: #eef6ff;
    margin: 10px 0 16px 0;
">
  <div style="font-size: 18px; font-weight: 700;">🔒 Confidentialité</div>
  <div style="margin-top:6px; font-size: 14px;">
    • Ton PDF n'est <b>pas stocké</b> par l'application.<br/>
    • Le traitement se fait <b>pendant l'analyse</b>, puis c'est terminé.<br/>
    • Si le PDF est un scan (image), l'app bascule automatiquement en OCR.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

DEBUG = st.checkbox("Mode debug", value=False)

# ------------------------------------------------------------
# Paiement (Stripe) — mode SaaS 7,50 €
# ------------------------------------------------------------
PRICE_EUR = 7.50
PAYMENT_LINK = os.getenv("STRIPE_PAYMENT_LINK", "").strip()  # ex: https://buy.stripe.com/...
ALLOW_NO_PAYMENT = os.getenv("ALLOW_NO_PAYMENT", "false").lower() == "true"

# ------------------------------------------------------------
# OPTION (recommandé en prod) : Webhook Stripe (anti-fraude 'béton')
#
# Le principe :
# 1) Stripe appelle ton serveur (endpoint webhook) sur checkout.session.completed
# 2) Ton serveur enregistre en base : session_id -> credit=1 (non consommé)
# 3) L'app Streamlit appelle ton serveur pour vérifier/consommer un crédit
#
# Dans Streamlit (ici), tu peux remplacer is_payment_ok() + session_state par :
#   - GET  /api/credits?session_id=...  -> {credits: 1}
#   - POST /api/consume?session_id=... -> {ok: true}
#
# Avantage : 1 paiement = 1 analyse même si l'utilisateur change de navigateur/appareil.
# ------------------------------------------------------------

def _get_query_param(name: str):
    """Compat Streamlit: query_params (>=1.32) ou experimental_get_query_params (ancien)."""
    if hasattr(st, "query_params"):
        return st.query_params.get(name)
    qp = st.experimental_get_query_params()
    v = qp.get(name)
    if isinstance(v, list):
        return v[0] if v else None
    return v

def is_payment_ok() -> tuple[bool, str]:
    if ALLOW_NO_PAYMENT:
        return True, "bypass"
    session_id = _get_query_param("session_id") or _get_query_param("checkout_session_id")
    if not session_id:
        return False, "missing_session_id"
    secret_key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not secret_key:
        return False, "missing_STRIPE_SECRET_KEY"
    try:
        import stripe  # lazy import
        stripe.api_key = secret_key
        s = stripe.checkout.Session.retrieve(session_id)
        paid = (getattr(s, "payment_status", None) == "paid") and (getattr(s, "status", None) in ("complete", "paid", None))
        return bool(paid), "paid" if paid else "not_paid"
    except Exception as e:
        return False, f"stripe_error:{type(e).__name__}"

# Paiement Stripe : vérifié plus tard (après pré-analyse)
# ------------------------------------------------------------
# Crédit d'analyse : 1 paiement = 1 analyse
#
# Remarque : Streamlit relance le script à chaque interaction.
# On 'consomme' le paiement côté session navigateur dès qu'on démarre l'analyse.
# Pour une protection 'béton' multi-navigateurs/appareils, voir la section Webhook Stripe plus bas.
# ------------------------------------------------------------
def _get_session_id_for_credit():
    return _get_query_param("session_id") or _get_query_param("checkout_session_id")


# ------------------------------------------------------------
# Crédit d'analyse : 1 paiement = 1 analyse (anti-refresh)
#
# Sur Render Free : pas de disque persistant → on stocke dans /tmp (survit aux refresh,
# mais pas aux redémarrages/redeploy). Quand tu passeras en Starter avec disque (/data),
# ce stockage deviendra persistant automatiquement.
# ------------------------------------------------------------
import sqlite3
from contextlib import closing

def _db_path() -> str:
    # si un disque persistant Render est présent, on l'utilise
    if os.path.isdir("/data"):
        return os.path.join("/data", "quadra_credits.db")
    return os.getenv("DB_PATH", "/tmp/quadra_credits.db")

def _db():
    con = sqlite3.connect(_db_path(), check_same_thread=False)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS consumed_sessions (
            session_id TEXT PRIMARY KEY,
            consumed_at TEXT NOT NULL
        )
        """
    )
    con.commit()
    return con

def credit_is_consumed(session_id: str) -> bool:
    if not session_id:
        return False
    con = _db()
    with closing(con.cursor()) as cur:
        cur.execute("SELECT 1 FROM consumed_sessions WHERE session_id=?", (session_id,))
        return cur.fetchone() is not None

def credit_consume(session_id: str) -> bool:
    """Retourne True si on a consommé le crédit, False si déjà consommé."""
    if not session_id:
        return False
    con = _db()
    try:
        con.execute(
            "INSERT INTO consumed_sessions(session_id, consumed_at) VALUES(?, datetime('now'))",
            (session_id,),
        )
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False


# Conserve un petit état côté session pour l'UI (non-sécurité).
if "analysis_credit_used_for" not in st.session_state:
    st.session_state.analysis_credit_used_for = None

_sid = _get_session_id_for_credit()
# si l'utilisateur revient avec un nouveau session_id payé, on ré-autorise l'UI
if _sid and (st.session_state.analysis_credit_used_for is not None) and (st.session_state.analysis_credit_used_for != _sid):
    st.session_state.analysis_credit_used_for = None
# Options d'analyse (après paiement)
# ------------------------------------------------------------
OCR_FORCE = st.checkbox("Forcer l'OCR (si PDF image)", value=False)
DPI = st.slider("Qualité OCR (DPI)", 150, 350, 250, 50)
uploaded = st.file_uploader("Dépose ton bulletin de salaire (PDF)", type=["pdf"], key="unique_file_uploader_2")


# Si Tesseract n'est pas trouvé sur Windows, décommente et adapte :
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ------------------------------------------------------------
# Nettoyage + montants
# ------------------------------------------------------------
def fix_doubled_letters(s: str) -> str:
    """Supprime les lettres doublées consécutives (SSiirreett -> Siret)."""
    result = []
    i = 0
    while i < len(s):
        ch = s[i]
        result.append(ch)
        if i + 1 < len(s) and s[i + 1] == ch and ch.isalpha():
            i += 2
        else:
            i += 1
    return "".join(result)


def dedouble_digits_if_all_pairs(token: str) -> str:
    """
    Dédouble un token uniquement si :
    - seulement des chiffres
    - longueur paire
    - chaque paire est identique (ex: 22002255 -> 2025)
    """
    if not token.isdigit() or len(token) % 2 != 0:
        return token
    out = []
    for i in range(0, len(token), 2):
        if token[i] != token[i + 1]:
            return token
        out.append(token[i])
    return "".join(out)


def normalize_doubled_digits_in_dates(text: str) -> str:
    """
    Corrige surtout les dates/années quand les chiffres sont doublés.
    Ex: 0011//1122//22002255 -> 01/12/2025
    """
    t = text

    def repl_date(m):
        a = dedouble_digits_if_all_pairs(m.group(1))
        b = dedouble_digits_if_all_pairs(m.group(2))
        c = dedouble_digits_if_all_pairs(m.group(3))
        return f"{a}/{b}/{c}"

    t = re.sub(r"\b(\d{2,4})[\/\-]{1,2}(\d{2,4})[\/\-]{1,2}(\d{4,8})\b", repl_date, t)
    t = re.sub(r"\b\d{8}\b", lambda m: dedouble_digits_if_all_pairs(m.group(0)), t)
    return t


def norm_spaces(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


def to_float_fr(s: str):
    if s is None:
        return None
    s = s.replace("\xa0", " ").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_amounts_money(line: str):
    raw = re.findall(r"\d[\d\s]*[.,]\d{2}", line)
    vals = []
    for r in raw:
        v = to_float_fr(r)
        if v is not None:
            vals.append(v)
    return vals


def extract_last_amount_money(line: str):
    vals = extract_amounts_money(line)
    return vals[-1] if vals else None


def find_last_line_with_amount(text: str, include_patterns, exclude_patterns=None):
    exclude_patterns = exclude_patterns or []
    best_line = None
    best_val = None
    for line in (text or "").splitlines():
        low = line.lower()
        if not any(re.search(p, low) for p in include_patterns):
            continue
        if any(re.search(p, low) for p in exclude_patterns):
            continue
        if not re.search(r"\d", line):
            continue
        v = extract_last_amount_money(line)
        if v is not None:
            best_line = line
            best_val = v
    return best_val, best_line


def eur(v):
    if v is None:
        return "-"
    s = f"{v:,.2f}".replace(",", " ").replace(".", ",")
    return f"{s} EUR"


# ------------------------------------------------------------
# PDF text + OCR auto
# ------------------------------------------------------------
def pdf_to_page_images(file, dpi=250):
    file.seek(0)
    images = []
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            images.append(p.to_image(resolution=dpi).original)
    return images


def ocr_pdf_to_text(file, dpi=250, lang="fra") -> str:
    imgs = pdf_to_page_images(file, dpi=dpi)
    out = []
    for im in imgs:
        out.append(pytesseract.image_to_string(im, lang=lang))
    return "\n".join(out)


def extract_text_auto(file, dpi=250, force_ocr=False):
    # IMPORTANT: le fichier est relu plusieurs fois (pdfplumber, conversion images, OCR)
    # => toujours remettre le curseur au début avant chaque lecture
    file.seek(0)
    classic = ""
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            classic += (p.extract_text() or "") + "\n"
    classic = norm_spaces(normalize_doubled_digits_in_dates(fix_doubled_letters(classic)))

    file.seek(0)
    images = pdf_to_page_images(file, dpi=dpi)

    if force_ocr or len(classic) < 80:
        file.seek(0)
        ocr_txt = ocr_pdf_to_text(file, dpi=dpi, lang="fra")
        ocr_txt = norm_spaces(ocr_txt)
        return ocr_txt, True, images

    return classic, False, images


# ------------------------------------------------------------
# Détection format
# ------------------------------------------------------------
def detect_format(text: str):
    t = (text or "").lower()
    quadra_hits = ["total des retenues", "total retenues", "quadra", "cegid"]
    silae_hits = ["coût global", "cout global", "cotisations et contributions", "silae"]
    score_quadra = sum(1 for k in quadra_hits if k in t)
    score_silae = sum(1 for k in silae_hits if k in t)

    if score_silae > score_quadra and score_silae >= 1:
        return "SILAE", {"score_silae": score_silae, "score_quadra": score_quadra}
    if score_quadra >= 1:
        return "QUADRA", {"score_silae": score_silae, "score_quadra": score_quadra}
    return "INCONNU", {"score_silae": score_silae, "score_quadra": score_quadra}



# ------------------------------------------------------------
# Validation du PDF (sécurité produit)
# ------------------------------------------------------------
PAYSPLIP_KEYWORDS = [
    "bulletin", "paie", "salaire", "salaire brut", "net", "net à payer", "net a payer",
    "cotisation", "cotisations", "csg", "crds", "prélèvement à la source", "prelevement a la source",
    "pas", "matricule", "salarié", "salarie",
]

def is_likely_payslip(text: str) -> tuple[bool, dict]:
    """
    Heuristique simple pour bloquer tout document qui n'est pas un bulletin de salaire.
    Retourne (ok, dbg).
    """
    t = (text or "").lower()
    hits = {k: (k in t) for k in PAYSPLIP_KEYWORDS}
    score = sum(1 for v in hits.values() if v)

    # Signaux "forts" : présence d'au moins 2 de ces 3 thèmes
    strong = 0
    if ("salaire brut" in t) or re.search(r"\bbrut\b", t):
        strong += 1
    if re.search(r"net\s+(pay[ée]|paye|a\s+payer|à\s+payer)", t):
        strong += 1
    if ("cotisation" in t) or ("csg" in t) or ("crds" in t):
        strong += 1

    ok = (score >= 4) and (strong >= 2)
    return ok, {"score": score, "strong": strong, "hits": [k for k, v in hits.items() if v]}

def _clean_person_token(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[\t]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ \-']", "", s)
    return s.strip()

def extract_employee_id(text: str):
    """
    Tente d'extraire l'identité du salarié pour autoriser un bulletin sur 2 pages
    uniquement si c'est le même salarié et la même période.
    Retourne un identifiant normalisé (ex: "DUPONT JEAN") ou None.
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    head = lines[:60]

    # 1) "Salarié : NOM Prénom"
    for l in head:
        m = re.search(r"(salari[ée]|salarie|employ[ée]|employe)\s*[:\-]\s*(.+)$", l, re.IGNORECASE)
        if m:
            name = _clean_person_token(m.group(2))
            if len(name) >= 5:
                return name.upper()

    # 2) "Nom : X" + "Prénom : Y"
    nom = prenom = None
    for l in head:
        m = re.search(r"\bnom\s*[:\-]\s*(.+)$", l, re.IGNORECASE)
        if m:
            nom = _clean_person_token(m.group(1))
        m = re.search(r"\bpr[ée]nom\s*[:\-]\s*(.+)$", l, re.IGNORECASE)
        if m:
            prenom = _clean_person_token(m.group(1))
    if nom and prenom:
        return f"{nom} {prenom}".upper()

    # 3) fallback : ligne "MAT / Matricule" suivie d'un nom (rare mais utile)
    for i, l in enumerate(head):
        if re.search(r"\bmatricule\b", l, re.IGNORECASE):
            # parfois le nom est sur la même ligne après le matricule
            after = re.split(r"\bmatricule\b", l, flags=re.IGNORECASE)[-1]
            after = _clean_person_token(after)
            if len(after) >= 5:
                return after.upper()
            # sinon sur la ligne suivante
            if i + 1 < len(head):
                cand = _clean_person_token(head[i + 1])
                if len(cand) >= 5:
                    return cand.upper()

    return None

def extract_text_auto_per_page(file, dpi=250, force_ocr=False):
    """
    Version 'par page' de l'extraction :
    - texte PDF classique par page
    - OCR seulement si page quasi vide (ou force_ocr)
    Retourne (all_text, used_ocr_any, images, page_texts, page_used_ocr_flags)
    """
    file.seek(0)
    images = pdf_to_page_images(file, dpi=dpi)

    page_texts = []
    page_ocr = []
    file.seek(0)
    with pdfplumber.open(file) as pdf:
        for i, p in enumerate(pdf.pages):
            t = p.extract_text() or ""
            t = norm_spaces(normalize_doubled_digits_in_dates(fix_doubled_letters(t)))
            page_texts.append(t)
            page_ocr.append(False)

    # OCR page par page si besoin
    for i, t in enumerate(page_texts):
        if force_ocr or len(t) < 40:
            try:
                ocr_t = norm_spaces(pytesseract.image_to_string(images[i], lang="fra"))
            except Exception:
                ocr_t = ""
            if len(ocr_t) > len(t):
                page_texts[i] = ocr_t
            page_ocr[i] = True

    all_text = "\n".join(page_texts).strip()
    used_ocr_any = any(page_ocr)
    return all_text, used_ocr_any, images, page_texts, page_ocr

def validate_uploaded_pdf(page_texts: list[str]) -> tuple[bool, str, dict]:
    """
    Règles demandées :
    - Bloquer tout document qui n'est pas un bulletin de salaire
    - Bloquer un PDF multi-pages, SAUF si c'est un bulletin (même salarié, même période) sur 2 pages
    """
    n = len(page_texts or [])
    all_text = "\n".join(page_texts or [])

    # 1) bulletin de salaire ?
    ok_ps, dbg_ps = is_likely_payslip(all_text)
    fmt, fmt_dbg = detect_format(all_text)

    if (not ok_ps) or (fmt == "INCONNU"):
        return (
            False,
            "🔒 Document refusé : ce PDF ne ressemble pas à un **bulletin de salaire** (ou format non reconnu).",
            {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n},
        )

    # 2) pages autorisées
    if n <= 1:
        return True, "OK", {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n}

    if n > 2:
        return (
            False,
            "🔒 Document refusé : PDF **multi-pages** (>2) non accepté. Dépose uniquement le bulletin concerné.",
            {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n},
        )

    # 3) cas 2 pages : même salarié + même période
    p1, p2 = page_texts[0], page_texts[1]

    period_all, _ = extract_period(all_text)
    period_1, _ = extract_period(p1)
    period_2, _ = extract_period(p2)

    # comparaison robuste (mois+année) pour éviter les faux refus dus à des libellés différents
    period_ref = period_1 or period_all
    key_ref = period_key(period_ref)
    key_p2 = period_key(period_2)

    if not key_ref:
        return (
            False,
            "🔒 Document refusé : bulletin sur 2 pages mais **période** illisible (je ne peux pas vérifier que c'est la même).",
            {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n, "periods": [period_1, period_2, period_all], "period_keys": [key_ref, key_p2]},
        )

    # Si la page 2 n'a pas de période, on accepte (beaucoup de bulletins n'affichent la période que sur la page 1).
    if key_p2 and (key_p2 != key_ref):
        return (
            False,
            "🔒 Document refusé : bulletin sur 2 pages mais **période différente** entre les pages.",
            {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n, "periods": [period_1, period_2, period_all], "period_keys": [key_ref, key_p2]},
        )

    emp_1 = extract_employee_id(p1) or extract_employee_id(all_text)
    emp_2 = extract_employee_id(p2) or extract_employee_id(all_text)

    if not emp_1:
        return (
            False,
            "🔒 Document refusé : bulletin sur 2 pages mais **salarié** illisible (je ne peux pas vérifier que c'est le même).",
            {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n, "employee": [emp_1, emp_2]},
        )

    # sur la 2e page, le nom n'est pas toujours 'labellisé' => vérif souple :
    # - soit extraction OK et identique
    # - soit le NOM (token principal) apparaît dans le texte de la page 2
    if emp_2 and emp_2 != emp_1:
        return (
            False,
            "🔒 Document refusé : bulletin sur 2 pages mais **salarié différent** entre les pages.",
            {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n, "employee": [emp_1, emp_2]},
        )

    last_token = emp_1.split()[0] if emp_1 else ""
    if last_token and (last_token.lower() not in (p2 or "").lower()) and (emp_2 is None):
        return (
            False,
            "🔒 Document refusé : bulletin sur 2 pages mais je ne retrouve pas le **même salarié** sur la page 2.",
            {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n, "employee": [emp_1, emp_2], "period_ref": period_ref},
        )

    return True, "OK", {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n, "employee": emp_1, "period_ref": period_ref}


# ------------------------------------------------------------
# Période
# ------------------------------------------------------------
MONTHS = {
    "janvier": "Janvier",
    "février": "Février",
    "fevrier": "Février",
    "mars": "Mars",
    "avril": "Avril",
    "mai": "Mai",
    "juin": "Juin",
    "juillet": "Juillet",
    "août": "Août",
    "aout": "Août",
    "septembre": "Septembre",
    "octobre": "Octobre",
    "novembre": "Novembre",
    "décembre": "Décembre",
    "decembre": "Décembre",
}



def period_key(period_str: str):
    """Retourne une clé canonique (année, mois) pour comparer des périodes de bulletin.

    Objectif: éviter les faux refus quand une page écrit 'Du 01/11/2025 au 30/11/2025'
    et l'autre 'Novembre 2025' ou '11/2025'.
    Renvoie None si on ne peut pas déterminer un mois+année.
    """
    if not period_str:
        return None
    s = norm_spaces(period_str).lower()

    # mois en français
    mois = {
        "janvier": 1, "janv": 1,
        "février": 2, "fevrier": 2, "févr": 2, "fevr": 2,
        "mars": 3,
        "avril": 4, "avr": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7, "juil": 7,
        "août": 8, "aout": 8,
        "septembre": 9, "sept": 9,
        "octobre": 10, "oct": 10,
        "novembre": 11, "nov": 11,
        "décembre": 12, "decembre": 12, "déc": 12, "dec": 12,
    }

    # cas: "du 01/11/2025 au 30/11/2025"
    m = re.search(r"\b(\d{2})[\/\-](\d{2})[\/\-](\d{4})\b.*?\b(\d{2})[\/\-](\d{2})[\/\-](\d{4})\b", s)
    if m:
        # on prend le mois/année du début et de fin; si identiques => ok; sinon on renvoie un tuple étendu
        m1, y1 = int(m.group(2)), int(m.group(3))
        m2, y2 = int(m.group(5)), int(m.group(6))
        if (m1, y1) == (m2, y2):
            return (y1, m1)
        # période chevauche deux mois : rare pour bulletin; on renvoie une clé range pour comparaison stricte
        return (y1, m1, y2, m2)

    # cas: "11/2025" ou "11-2025"
    m = re.search(r"\b(0?[1-9]|1[0-2])[\/\-](20\d{2})\b", s)
    if m:
        return (int(m.group(2)), int(m.group(1)))

    # cas: "novembre 2025"
    for name, num in mois.items():
        if re.search(rf"\b{name}\b", s):
            y = None
            my = re.search(r"\b(20\d{2})\b", s)
            if my:
                y = int(my.group(1))
            if y:
                return (y, num)

    return None


def extract_period(text: str):
    for line in (text or "").splitlines():
        low = line.lower()
        if "paiement" in low:
            continue
        m = re.search(
            r"\bdu\s*[:\-]?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4}).*?\bau\s*[:\-]?\s*(\d{2}[\/\-]\d{2}[\/\-]\d{4})\b",
            low,
            re.IGNORECASE,
        )
        if m:
            return f"Du {m.group(1)} au {m.group(2)}", line

    for line in (text or "").splitlines():
        low = line.lower()
        if "paiement" in low:
            continue
        if "période" in low or "periode" in low:
            m = re.search(
                r"(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre)\s+(\d{4})",
                low,
            )
            if m:
                mois = MONTHS.get(m.group(1), m.group(1).capitalize())
                return f"{mois} {m.group(2)}", line

    return None, None


# ------------------------------------------------------------
# QUADRA — charges, acompte, total versé employeur, CP
# ------------------------------------------------------------
def extract_charges_quadra(text: str):
    charges_sal = None
    charges_pat = None
    line_used = None
    method = None
    for line in (text or "").splitlines():
        low = line.lower()
        if "total des retenues" in low or "total retenues" in low:
            vals = extract_amounts_money(line)
            if len(vals) >= 2:
                charges_sal = vals[-2]
                charges_pat = vals[-1]
                line_used = line.strip()
                method = "total_retenues_two_amounts"
    return charges_sal, charges_pat, line_used, method


def extract_csg_non_deductible_total(text: str):
    """
    QUADRA : CSG/CRDS non déductible peut être sur plusieurs lignes.
    Certaines lignes contiennent un autre montant parasite en fin de ligne (ex: 176.67).
    Stratégie robuste : repérer un taux (%) puis prendre le montant qui suit immédiatement ce taux.
    Retour : (total, lines_used, debug_picks)
    """
    total = 0.0
    lines_used = []
    debug_picks = []

    # exemple ligne: "... 2456.27 2.9000 71.23"
    # ou "... 522.28 9.7000 50.66 176.67"
    # => on veut le montant juste après le taux.
    for line in text.splitlines():
        low = line.lower()
        if "csg" not in low:
            continue
        if ("non déduct" not in low) and ("non deduct" not in low):
            continue

        # extraire des "tokens" numériques dans l'ordre
        toks = re.findall(r"\d+(?:[.,]\d+)?", line)
        if not toks:
            continue

        # convertir tokens en float
        nums = []
        for t in toks:
            v = to_float_fr(t.replace(".", ".").replace(",", "."))
            if v is not None:
                nums.append(v)

        if len(nums) < 2:
            continue

        picked = None

        # chercher un "taux" plausible (2.9 ; 6.8 ; 9.7 etc.)
        # Dans Quadra il apparaît souvent comme 2.9000 / 9.7000
        for i, v in enumerate(nums):
            if 0.1 <= v <= 20.0:  # taux plausible
                # on prend le montant qui suit immédiatement le taux
                if i + 1 < len(nums):
                    cand = nums[i + 1]
                    # montant plausible de CSG : généralement < 1000
                    if 0 <= cand <= 2000:
                        picked = cand
                        break

        # fallback si pas de taux trouvé : prendre le plus petit montant de la ligne (évite le cumul)
        if picked is None:
            # on évite les bases type 2456.27, on garde plutôt un montant < 2000
            small = [v for v in nums if 0 <= v <= 2000]
            if small:
                picked = min(small)

        if picked is not None:
            total += picked
            lines_used.append(line)
            debug_picks.append((line, picked, nums))

    if not lines_used:
        return None, None

    return round(total, 2), lines_used



def extract_acompte(text: str, net_paye: float | None = None, brut: float | None = None):
    """
    Acompte / avance / déjà versé :
    - évite les faux positifs type '2025' collé à '500,00' -> 2 025 500,00
    - filtre par plausibilité (mensuel)
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    best = None
    best_line = None

    if net_paye is not None and net_paye > 0:
        max_ok = net_paye * 1.2
    elif brut is not None and brut > 0:
        max_ok = brut * 1.2
    else:
        max_ok = 5000

    for line in lines:
        low = line.lower()
        if not (("acompte" in low) or ("avance" in low) or ("déjà vers" in low) or ("deja vers" in low)):
            continue

        vals = extract_amounts_money(line)
        if not vals:
            continue

        has_year = bool(re.search(r"\b20\d{2}\b", line))
        for v in vals:
            if v <= 0:
                continue
            if v > max_ok:
                continue
            if has_year and v > 2000:
                continue
            best = v
            best_line = line

    return best, best_line


def extract_total_verse_employeur_quadra(text: str, brut: float | None, net_paye: float | None):
    """
    QUADRA (cas réel) :
      - 'Total' sur une ligne
      - 'Versé employeur' en dessous (souvent sans montant sur la même ligne)
      - montant peut apparaître sur i, i+1, i+2, i+3, i+4 selon extraction
    Particularité 2 pages : la page 1 peut contenir 'Total'/'Versé employeur' SANS montant,
    le vrai montant peut être sur la page 2. Donc : on scanne TOUTES les occurrences et
    on garde le meilleur montant plausible.
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    low = [l.lower() for l in lines]

    def plausible(v: float) -> bool:
        if v is None or v <= 0:
            return False
        # bornes "mensuel" très larges mais utiles contre les faux positifs
        if net_paye is not None and v < net_paye:
            return False
        if brut is not None and v < brut * 0.80:
            return False
        if brut is not None and v > brut * 12.0:
            return False
        if v < 200:
            return False
        if v > 100000:
            return False
        return True

    best_v = None
    best_line = None

    def consider_window(window, ctx_prefix=""):
        nonlocal best_v, best_line
        for wl in window:
            for v in extract_amounts_money(wl):
                if plausible(v):
                    if (best_v is None) or (v > best_v):
                        best_v = v
                        best_line = f"{ctx_prefix}{wl}".strip()

    # 1) Couplage "Total" puis "Versé employeur" (peut apparaître plusieurs fois)
    for i in range(len(lines) - 1):
        if re.search(r"total", low[i]) and ("versé employeur" in low[i + 1] or "verse employeur" in low[i + 1]):
            start = max(0, i - 1)
            end = min(len(lines), i + 8)  # fenêtre plus large
            window = lines[start:end]
            consider_window(window, ctx_prefix=f"{lines[i]} -> {lines[i+1]} | ")

    # 2) Fallback : toute ligne contenant "versé employeur"
    for i, ll in enumerate(low):
        if "versé employeur" in ll or "verse employeur" in ll:
            start = max(0, i - 3)
            end = min(len(lines), i + 10)
            window = lines[start:end]
            consider_window(window, ctx_prefix=f"{lines[i]} | ")

    if best_v is None:
        return None, None
    return round(best_v, 2), best_line

def extract_cp_quadra(text: str):
    """
    CP QUADRA : cherche une zone CP/congés et prend 'Solde' (N-1 / N).
    """
    def parse_cp_token(tok: str):
        t = tok.strip().replace(",", ".")
        t = re.sub(r"[^0-9.]", "", t)

        if re.fullmatch(r"\d+\.\d{2}", t):
            v = float(t)
            if 0 <= v <= 60:
                return round(v, 2)

        digits = re.sub(r"\D", "", t)
        if len(digits) >= 4 and len(digits) % 2 == 0:
            ok = True
            out = []
            for i2 in range(0, len(digits), 2):
                if digits[i2] != digits[i2 + 1]:
                    ok = False
                    break
                out.append(digits[i2])
            if ok:
                dd = "".join(out)
                if len(dd) == 1:
                    v = float(dd)
                elif len(dd) == 2:
                    v = float(dd[0] + "." + dd[1])
                else:
                    v = float(dd[:-2] + "." + dd[-2:])
                if 0 <= v <= 60:
                    return round(v, 2)

        try:
            v = float(t)
            if 0 <= v <= 60:
                return round(v, 2)
        except:
            pass
        return None

    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    start_idx = None
    for i, l in enumerate(lines):
        low = l.lower()
        if "cp" in low or "cong" in low:
            start_idx = i
            break

    if start_idx is None:
        return {"cp_n1": None, "cp_n": None, "cp_total": None, "cp_method": "not_found", "cp_solde_line": None}

    window = lines[start_idx:start_idx + 40]
    solde_line = None
    for l in window:
        if "solde" in l.lower():
            s = re.sub(r"[ ]+", " ", l.replace("..", ".").replace("::", ":").replace("//", "/")).strip()
            solde_line = s
            break

    if solde_line is None:
        return {"cp_n1": None, "cp_n": None, "cp_total": None, "cp_method": "no_solde_line", "cp_solde_line": None}

    raw_tokens = re.findall(r"\d+(?:[.,]\d{1,4})?", solde_line)
    vals = []
    for tok in raw_tokens:
        v = parse_cp_token(tok)
        if v is not None:
            vals.append(v)

    cp_n1 = None
    cp_n = None
    if len(vals) >= 2:
        cp_n1, cp_n = vals[0], vals[1]
        method = "solde_two_vals"
    elif len(vals) == 1:
        method = "solde_one_val"
    else:
        method = "solde_no_numbers"

    cp_total = round((cp_n1 or 0.0) + (cp_n or 0.0), 2) if (cp_n1 is not None or cp_n is not None) else None
    return {"cp_n1": cp_n1, "cp_n": cp_n, "cp_total": cp_total, "cp_method": method, "cp_solde_line": solde_line}


# ------------------------------------------------------------
# SILAE — coût global (mensuel) via OCR zone, CP via OCR bas gauche
# ------------------------------------------------------------
def extract_cout_global_fast(page_img):
    """Version rapide (souvent) : OCR d'une zone en haut de page puis recherche 'coût global'.
    Fallback vers extract_cout_global_from_image() si échec.
    """
    try:
        W, H = page_img.width, page_img.height
        crop = page_img.crop((0, 0, W, int(H * 0.40)))
        zone_text = norm_spaces(pytesseract.image_to_string(crop, lang="fra"))
        lines = [l.strip() for l in zone_text.splitlines() if l.strip()]
        for i, l in enumerate(lines):
            ll = l.lower()
            if ("coût global" in ll) or ("cout global" in ll):
                chunk = " ".join(lines[i:i+4])
                vals = extract_amounts_money(chunk)
                vals_ok = [v for v in vals if 0 < v < 20000]
                if vals_ok:
                    return vals_ok[0], f"fast_ocr: {chunk[:200]}"
        return None, f"fast_ocr_nohit: {zone_text[:220]}"
    except Exception as e:
        return None, f"fast_ocr_error:{type(e).__name__}"

def extract_cout_global_from_image(page_img):
    """
    Repère le titre "Coût global" et lit le montant mensuel juste en dessous.
    """
    data = pytesseract.image_to_data(page_img, lang="fra", output_type=Output.DICT)
    n = len(data["text"])
    tokens = []
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        tokens.append({
            "t": txt.lower(),
            "x": int(data["left"][i]),
            "y": int(data["top"][i]),
            "w": int(data["width"][i]),
            "h": int(data["height"][i]),
        })

    bbox = None
    for a in tokens:
        if a["t"] not in ("coût", "cout", "coût,", "cout,"):
            continue
        for b in tokens:
            if b["t"] != "global":
                continue
            same_line = abs(a["y"] - b["y"]) <= max(a["h"], b["h"]) * 1.3
            close_x = 0 <= (b["x"] - a["x"]) <= 300
            if same_line and close_x:
                x1 = min(a["x"], b["x"])
                y1 = min(a["y"], b["y"])
                x2 = max(a["x"] + a["w"], b["x"] + b["w"])
                y2 = max(a["y"] + a["h"], b["y"] + b["h"])
                bbox = (x1, y1, x2, y2)
                break
        if bbox:
            break

    if not bbox:
        return None, "(titre 'Coût global' non trouvé)"

    x1, y1, x2, y2 = bbox
    pad_x = 20
    pad_y = 6

    # zone en dessous du titre
    crop_left = max(0, x1 - pad_x)
    crop_top = min(page_img.height - 1, y2 + pad_y)
    crop_right = min(page_img.width, crop_left + 380)
    crop_bottom = min(page_img.height, crop_top + 120)

    crop = page_img.crop((crop_left, crop_top, crop_right, crop_bottom))
    zone_text = norm_spaces(pytesseract.image_to_string(crop, lang="fra"))

    vals = extract_amounts_money(zone_text)
    vals_ok = [v for v in vals if 0 < v < 10000]
    if not vals_ok:
        return None, f"(zone coût global mais aucun montant plausible) zone_text={zone_text[:180]}"

    return vals_ok[0], f"zone_cost_global_text: {zone_text[:180]}"

def extract_acompte(text: str, net_paye: float | None = None, brut: float | None = None):
    """
    Acompte / avance / déjà versé :
    - corrige le cas SILAE où l'année (2025 / '2 025') se colle au montant -> 2 025 500,00
    - filtre par plausibilité (mensuel)
    """
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]

    # Bornes plausibles (mensuel)
    if net_paye is not None and net_paye > 0:
        max_ok = net_paye * 1.5  # un acompte peut dépasser le net payé exceptionnellement (rare), on laisse 1.5
    elif brut is not None and brut > 0:
        max_ok = brut * 1.5
    else:
        max_ok = 6000

    def pick_plausible(vals):
        vals = [v for v in vals if v is not None and 0 < v <= max_ok]
        if not vals:
            return None
        # en pratique l'acompte est souvent le plus grand montant plausible sur la ligne (ex: 500,00)
        return max(vals)

    def sanitize_year_glue(s: str) -> str:
        # enlève 2025, 2024...
        s2 = re.sub(r"\b20\d{2}\b", " ", s)
        # enlève année "cassée" type "2 025", "2 024"...
        s2 = re.sub(r"\b2\s+0\d{2}\b", " ", s2)
        return s2

    best = None
    best_line = None

    for line in lines:
        low = line.lower()
        if not (("acompte" in low) or ("avance" in low) or ("déjà vers" in low) or ("deja vers" in low)):
            continue

        # 1) extraction brute
        vals1 = extract_amounts_money(line)
        v = pick_plausible(vals1)

        # 2) si rien de plausible (souvent car l'année a été collée), on "décolle" l'année et on ré-extrait
        if v is None:
            line2 = sanitize_year_glue(line)
            vals2 = extract_amounts_money(line2)
            v = pick_plausible(vals2)

        if v is not None:
            best = v
            best_line = line

    return best, best_line



def extract_cp_from_image_silae(page_img):
    """
    SILAE : CP en bas à gauche, tableau N-1 / N, ligne 'Solde'.
    On snap à 0.25 pour éviter 19.06 vs 19.00.
    """
    def snap_cp(v: float | None):
        if v is None:
            return None
        return round(round(v / 0.25) * 0.25, 2)

    def extract_cp_numbers(line: str):
        raw = re.findall(r"\d+(?:[.,]\d{1,2})?", line)
        out = []
        for r in raw:
            v = to_float_fr(r)
            if v is not None:
                out.append(v)
        return out

    W, H = page_img.width, page_img.height
    left = 0
    top = int(H * 0.78)
    right = int(W * 0.45)
    bottom = H

    crop = page_img.crop((left, top, right, bottom))
    zone_text = norm_spaces(pytesseract.image_to_string(crop, lang="fra"))

    solde_line = None
    best_count = -1
    for line in zone_text.splitlines():
        ll = line.lower()
        if "solde" not in ll:
            continue
        nums = extract_cp_numbers(line)
        if len(nums) > best_count:
            best_count = len(nums)
            solde_line = line

    if not solde_line:
        return {"cp_n1": None, "cp_n": None, "cp_total": None, "cp_debug": zone_text[:260]}

    nums = extract_cp_numbers(solde_line)
    cp_n1 = snap_cp(nums[0]) if len(nums) >= 1 else None
    cp_n = snap_cp(nums[1]) if len(nums) >= 2 else None
    cp_total = round((cp_n1 or 0.0) + (cp_n or 0.0), 2) if (cp_n1 is not None or cp_n is not None) else None
    return {"cp_n1": cp_n1, "cp_n": cp_n, "cp_total": cp_total, "cp_debug": f"solde_line={solde_line}"}


# ------------------------------------------------------------
# Commentaires
# ------------------------------------------------------------


def extract_cp_from_text_any(text: str):
    """Essaie d'extraire les congés payés (solde N-1 / N) depuis un texte (OCR ou PDF).
    Retourne un dict compatible avec extract_cp_from_image_silae()."""
    text = text or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Cherche une ligne contenant 'Solde' (souvent la ligne clé)
    solde_idx = None
    for i, l in enumerate(lines):
        if "solde" in l.lower():
            solde_idx = i
            break

    if solde_idx is None:
        return {"cp_n1": None, "cp_n": None, "cp_total": None, "cp_method": "text_no_solde", "cp_solde_line": None}

    # Regroupe 1-2 lignes autour pour capter les chiffres
    chunk = " ".join(lines[max(0, solde_idx - 1): min(len(lines), solde_idx + 2)])
    vals = [v for v in extract_amounts_money(chunk) if 0 <= v <= 60]  # CP en jours/heures, rarement >60
    cp_n1 = None
    cp_n = None
    if len(vals) >= 2:
        cp_n1, cp_n = vals[0], vals[1]
        method = "text_solde_two_vals"
    elif len(vals) == 1:
        cp_n = vals[0]
        method = "text_solde_one_val"
    else:
        method = "text_solde_no_numbers"

    cp_total = round((cp_n1 or 0.0) + (cp_n or 0.0), 2) if (cp_n1 is not None or cp_n is not None) else None
    return {"cp_n1": cp_n1, "cp_n": cp_n, "cp_total": cp_total, "cp_method": method, "cp_solde_line": chunk[:200]}


def _choose_silae_page_index(page_images, page_texts):
    """Le bulletin SILAE peut faire 1 ou 2 pages.
    On essaie d'identifier la page qui contient le tableau avec 'Coût global'."""
    for i, t in enumerate(page_texts or []):
        tl = (t or "").lower()
        if "coût global" in tl or "cout global" in tl:
            return i

    if not page_images:
        return 0

    for i, im in enumerate(page_images):
        W, H = im.width, im.height
        band = im.crop((0, int(H * 0.70), W, H))
        try:
            txt = pytesseract.image_to_string(band, lang="fra", config="--psm 6")
        except Exception:
            txt = ""
        tl = (txt or "").lower()
        if "coût global" in tl or "cout global" in tl:
            return i

    return 0


def extract_silae_cost_and_cp(page_img, page_text: str | None = None):
    """Extraction optimisée SILAE (plus rapide que 2 OCR séparés).
    - tentative via texte (si présent)
    - OCR bande basse pour coût global + CP
    - fallback vers extract_cout_global_from_image / extract_cp_from_image_silae si besoin
    """
    debug = {}

    # 1) tentative via texte
    if page_text:
        tl = page_text.lower()
        if ("coût global" in tl) or ("cout global" in tl):
            for line in (page_text.splitlines() or []):
                ll = line.lower()
                if ("coût global" in ll) or ("cout global" in ll):
                    vals = [v for v in extract_amounts_money(line) if 0 < v < 50000]
                    if vals:
                        cout = vals[-1]
                        cp = extract_cp_from_text_any(page_text)
                        debug["mode"] = "text"
                        return cout, f"text: {line[:200]}", cp, debug

    # 2) OCR bande basse
    W, H = page_img.width, page_img.height
    bottom = page_img.crop((0, int(H * 0.62), W, H))
    ocr_text = norm_spaces(pytesseract.image_to_string(bottom, lang="fra", config="--psm 6"))
    debug["bottom_ocr_head"] = ocr_text[:220]

    cout = None
    cout_line = None
    for line in ocr_text.splitlines():
        ll = line.lower()
        if ("coût global" in ll) or ("cout global" in ll):
            vals = [v for v in extract_amounts_money(line) if 0 < v < 50000]
            if vals:
                cout = vals[-1]
                cout_line = line[:200]
                break

    cp = extract_cp_from_text_any(ocr_text)

    # fallback CP petite zone (si pas trouvé)
    if cp.get("cp_total") is None:
        cp = extract_cp_from_image_silae(page_img)
        debug["cp_fallback"] = True

    # fallback coût global précis
    if cout is None:
        cout, cout_line = extract_cout_global_from_image(page_img)
        debug["cout_fallback"] = True

    return cout, (cout_line or "bottom_ocr"), cp, debug


def build_comments(brut, charges_sal, csg_nd, pas, charges_pat, acompte):
    out = []

    # Acompte : prévenir tout de suite
    if acompte is not None and acompte > 0:
        out.append("• Acompte détecté : le 'net payé' correspond à ce qui arrive ce mois-ci, pas forcément au net total du bulletin.")

    # Indicateur stable côté salarié : prélèvements sur le brut
    if brut is not None and brut > 0:
        prelev = (charges_sal or 0.0) + (csg_nd or 0.0)
        ratio = round((prelev / brut) * 100, 1)
        out.append(f"• Prélèvements sur ton brut : {ratio}% (cotisations salariales + CSG non déductible).")

    # PAS
    if pas is not None:
        if pas == 0:
            out.append("• Impôt à 0 EUR : aujourd'hui, le fisc observe. Sans bouger. Moment rare.")
        else:
            out.append("• Le PAS est prélevé sur ton salaire, puis versé au fisc par l'employeur.")

    # CSG non déductible (humour + explication courte)
    if csg_nd is not None and csg_nd > 0:
        out.append("• La CSG non déductible : tu ne la touches pas, mais elle augmente ton net imposable. Plot twist fiscal.")

    # Charges patronales
    if charges_pat is not None and charges_pat > 0:
        out.append("• Ton employeur dépense plus que ce que tu touches. Ce n'est pas toi, c'est le système.")

    return out[:7]


# ------------------------------------------------------------
# PDF (carrés lisibles)
# ------------------------------------------------------------
def wrap_title_for_box(title: str, max_chars=18):
    words = title.split()
    lines = []
    cur = ""
    for w in words:
        if len(cur) + len(w) + (1 if cur else 0) > max_chars:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines[:2]


def step_box(c, x, y, w, h, title, value, fill_hex):
    c.setStrokeColor(colors.HexColor("#222222"))
    c.setFillColor(colors.HexColor(fill_hex))
    c.roundRect(x, y - h, w, h, 10, stroke=1, fill=1)

    # titre (haut)
    title_lines = wrap_title_for_box(title, max_chars=18)
    c.setFillColor(colors.HexColor("#111111"))
    c.setFont("Helvetica-Bold", 9 if len(title_lines) == 2 else 10)
    ty = y - 16
    for tl in title_lines:
        c.drawString(x + 10, ty, tl)
        ty -= 11

    # valeur (bas)
    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(x + w / 2, y - h + 22, value)


def arrow(c, x1, y1, x2, y2):
    c.setStrokeColor(colors.HexColor("#333333"))
    c.setLineWidth(1.5)
    c.line(x1, y1, x2, y2)

    dx = x2 - x1
    dy = y2 - y1
    L = (dx * dx + dy * dy) ** 0.5 or 1
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    ah = 6
    aw = 3
    ax, ay = x2, y2
    c.line(ax, ay, ax - ah * ux + aw * px, ay - ah * uy + aw * py)
    c.line(ax, ay, ax - ah * ux - aw * px, ay - ah * uy - aw * py)


def card(c, x, y, w, h, title, lines, fill_hex):
    c.setStrokeColor(colors.HexColor("#222222"))
    c.setFillColor(colors.HexColor(fill_hex))
    c.roundRect(x, y - h, w, h, 10, stroke=1, fill=1)

    c.setFillColor(colors.HexColor("#111111"))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(x + 10, y - 18, title)

    c.setFont("Helvetica", 9)
    yy = y - 36
    for ln in lines:
        c.drawString(x + 10, yy, ln)
        yy -= 12


def build_pdf(fields, comments, fmt_name):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    margin = 2 * cm
    x0 = margin
    y = H - margin

    c.setFont("Helvetica-Bold", 16)
    c.drawString(x0, y, "Résumé simplifié de ton bulletin de salaire")
    y -= 18

    c.setFont("Helvetica-Oblique", 10)
    c.setFillColor(colors.HexColor("#555555"))
    c.drawString(x0, y, "Document pédagogique - le bulletin officiel reste la référence")
    y -= 14
    c.drawString(x0, y, f"Format détecté : {fmt_name}")
    y -= 16

    c.setFillColor(colors.HexColor("#eef6ff"))
    c.setStrokeColor(colors.HexColor("#1f77b4"))
    c.roundRect(x0, y - 34, W - 2 * margin, 34, 8, stroke=1, fill=1)
    c.setFillColor(colors.HexColor("#0b3d62"))
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x0 + 10, y - 14, "Confidentialité :")
    c.setFont("Helvetica", 9)
    c.drawString(x0 + 110, y - 14, "aucun stockage du PDF - analyse ponctuelle.")
    y -= 48

    c.setFillColor(colors.HexColor("#555555"))
    c.setFont("Helvetica", 10)
    c.drawString(x0, y, f"Période : {fields.get('period') or '-'}")
    c.drawRightString(W - margin, y, f"Généré le {dt.datetime.now().strftime('%d/%m/%Y %H:%M')}")
    c.setFillColor(colors.black)
    y -= 18

    brut = fields.get("brut")
    net_paye = fields.get("net_paye")
    acompte = fields.get("acompte")
    net_ref = fields.get("net_reference")
    pas = fields.get("pas")
    charges_sal = fields.get("charges_sal")
    charges_pat = fields.get("charges_pat")
    csg_nd = fields.get("csg_non_deductible")
    organismes_total = fields.get("organismes_total")
    cout_total = fields.get("cout_total")
    cp = fields.get("cp", {}) or {}

    # 4 carrés
    path_top = y - 10
    path_h = 86
    gap = 10
    w_total = W - 2 * margin
    step_w = (w_total - 3 * gap) / 4
    step_h = path_h
    step_y = path_top

    x1 = x0
    x2 = x1 + step_w + gap
    x3 = x2 + step_w + gap
    x4 = x3 + step_w + gap

    step_box(c, x1, step_y, step_w, step_h, "1) Salaire brut", eur(brut), "#FFFFFF")
    step_box(c, x2, step_y, step_w, step_h, "2) Cotisations salariales", eur(charges_sal), "#F3E8FF")
    step_box(c, x3, step_y, step_w, step_h, "3) Impôt (PAS)", eur(pas), "#FFF4E5")
    step_box(c, x4, step_y, step_w, step_h, "4) Net payé", eur(net_paye), "#E8F7EE")

    mid_y = step_y - step_h / 2
    arrow(c, x1 + step_w, mid_y, x2, mid_y)
    arrow(c, x2 + step_w, mid_y, x3, mid_y)
    arrow(c, x3 + step_w, mid_y, x4, mid_y)

    y_after_path = step_y - step_h - 18
    c.setFont("Helvetica", 9)
    c.setFillColor(colors.HexColor("#555555"))
    # (Info brut/net supprimée : peut être trompeuse avec indemnités, remboursements, acompte…)

    cards_top = y_after_path - 18
    card_h = 125
    card_w = (w_total - 12) / 2
    left_x = x0
    right_x = x0 + card_w + 12

    emp_lines = [f"Il te verse : {eur(net_paye)}"]
    if acompte and acompte > 0:
        emp_lines.append(f"Acompte déjà versé : {eur(acompte)}")
        emp_lines.append(f"Net reconstitué : {eur(net_ref)}")
    emp_lines.append(f"Organismes sociaux (total) : {eur(organismes_total)}")
    emp_lines.append(f"Impôts (PAS) : {eur(pas)}")
    emp_lines.append(f"Coût total employeur : {eur(cout_total)}")

    card(c, left_x, cards_top, card_w, card_h, "Côté employeur", emp_lines, "#EAF2FF")

    if cp.get("cp_total") is not None:
        cp_lines = [
            f"CP N-1 (solde) : {cp.get('cp_n1', 0.0):.2f} j",
            f"CP N (solde) : {cp.get('cp_n', 0.0):.2f} j",
            f"Total : {cp.get('cp_total', 0.0):.2f} j",
            "Solde = jours disponibles.",
        ]
    else:
        cp_lines = ["Congés : non lisibles automatiquement sur ce PDF.", "Je préfère être honnête que créatif."]

    card(c, right_x, cards_top, card_w, card_h, "Congés payés (solde)", cp_lines, "#F3E8FF")

    # commentaires (il reste de l'espace, on garde aéré)
    y_comments = cards_top - card_h - 28
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.HexColor("#111111"))
    c.drawString(x0, y_comments, "Ce qu'il faut retenir (humour factuel)")
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.black)

    def wrap(s, max_chars=105):
        words = s.split()
        out = []
        cur = ""
        for w in words:
            if len(cur) + len(w) + 1 > max_chars:
                out.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            out.append(cur)
        return out

    yy = y_comments - 16
    for com in comments:
        for part in wrap(com):
            c.drawString(x0, yy, part)
            yy -= 12
        yy -= 6  # plus d'air

    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(colors.HexColor("#666666"))
    c.drawString(x0, 1.3 * cm, "Le bulletin officiel reste le document de référence.")
    c.setFillColor(colors.black)

    c.showPage()
    c.save()
    buf.seek(0)
    return buf


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
if uploaded is not None:
    st.success("PDF reçu ✅")

    import hashlib as _hashlib
    _pdf_bytes = uploaded.getvalue()
    _pdf_digest = _hashlib.sha256(_pdf_bytes).hexdigest()

    # On mémorise si la pré-analyse a déjà été faite pour CE fichier (anti re-run inutile)
    if "precheck_ok_digest" not in st.session_state:
        st.session_state.precheck_ok_digest = None

    # ------------------------------------------------------------
    # Étape 1 — Pré-analyse (gratuite) : lisibilité + validation "bulletin"
    # ------------------------------------------------------------
    st.markdown("### Étape 1 — Pré-analyse (lisibilité du bulletin)")
    st.caption("On vérifie que ton fichier ressemble bien à un bulletin (et qu'il est lisible), avant de te demander de payer.")

    if st.button("Lancer la pré-analyse", type="primary"):
        with st.status("Pré-analyse en cours…", expanded=False) as _pre:
            _pre.write("Lecture du PDF + extraction texte (OCR si besoin)…")
            file_obj = io.BytesIO(_pdf_bytes)
            text_all, used_ocr, page_images, page_texts, page_ocr_flags = extract_text_auto_per_page(
                file_obj, dpi=DPI, force_ocr=OCR_FORCE
            )

            _pre.write("Vérification du document…")
            ok_doc, msg_doc, doc_dbg = validate_uploaded_pdf(page_texts)
            if not ok_doc:
                st.error(msg_doc)
                if DEBUG:
                    st.json(doc_dbg)
                st.stop()

            fmt, fmt_dbg = detect_format(text_all)
            period, _ = extract_period(text_all)

            # "Analyse partielle" : on donne juste quelques infos si on les trouve
            brut, _ = find_last_line_with_amount(
                text_all,
                include_patterns=[r"salaire\s+brut", r"\bbrut\b"],
                exclude_patterns=[r"net", r"imposable", r"csg", r"crds"],
            )
            net_paye, _ = find_last_line_with_amount(
                text_all,
                include_patterns=[r"net\s+paye", r"net\s+pay[ée]", r"net\s+à\s+payer", r"net\s+a\s+payer"],
                exclude_patterns=[r"avant\s+imp", r"imposable"],
            )

            _pre.update(label="Pré-analyse terminée", state="complete")

        st.session_state.precheck_ok_digest = _pdf_digest

        st.success("✅ Bulletin lisible et valide.")
        st.write(f"• Format détecté : **{fmt}**")
        if period:
            st.write(f"• Période : **{period}**")
        st.write(f"• OCR utilisé : **{used_ocr}**")
        if brut is not None:
            st.write(f"• Brut (détecté) : **{eur(brut)}**")
        if net_paye is not None:
            st.write(f"• Net payé (détecté) : **{eur(net_paye)}**")

        st.info("➡️ Si tu veux la synthèse complète + PDF, passe au paiement juste en dessous.")

        if DEBUG:
            st.json({"fmt_dbg": fmt_dbg})
            with st.expander("Texte extrait (début)"):
                st.text((text_all or "")[:8000])

    # ------------------------------------------------------------
    # Étape 2 — Paiement (Stripe) : seulement après pré-analyse OK
    # ------------------------------------------------------------
    if st.session_state.precheck_ok_digest == _pdf_digest:

        st.markdown("### Étape 2 — Paiement (7,50 €)")
        paid_ok, paid_reason = is_payment_ok()

        if not paid_ok and not ALLOW_NO_PAYMENT:
            st.write("Pour lancer **l'analyse complète** (et générer la synthèse PDF), il faut régler **7,50 €**.")
            if PAYMENT_LINK:
                st.link_button("Payer 7,50 €", PAYMENT_LINK, type="primary")
                st.caption("Après paiement, Stripe te redirige ici avec un `session_id` dans l'URL.")
            else:
                st.error("Paiement non configuré : variable d'environnement STRIPE_PAYMENT_LINK manquante.")
            if DEBUG:
                st.info(f"[debug] paiement non validé: {paid_reason}")

        else:
            st.success("✅ Paiement validé (ou bypass activé).")
            st.markdown("### Étape 3 — Analyse complète + synthèse")
            st.caption("Maintenant on relance l'analyse complète (1 paiement = 1 analyse).")

            if st.button("Lancer l'analyse complète", type="primary"):

                _sid = _get_session_id_for_credit()

            if not _sid:
                            st.error("session_id manquant dans l'URL. Reviens depuis la page de succès Stripe (success_url).")
                            st.stop()

                        # Sécurité : on vérifie / consomme côté serveur (SQLite). Empêche le bypass au refresh.
            if credit_is_consumed(_sid):
                            st.error("🔒 Ce paiement a déjà été utilisé : **1 paiement = 1 analyse**.\n\n➡️ Pour analyser un autre bulletin, repasse par le paiement.")
                            st.stop()

                        # On consomme le crédit AVANT de lancer le travail lourd.
            if not credit_consume(_sid):
                            st.error("🔒 Ce paiement a déjà été utilisé : **1 paiement = 1 analyse**.\n\n➡️ Pour analyser un autre bulletin, repasse par le paiement.")
                            st.stop()

  # UI only
st.session_state.analysis_credit_used_for = _sid

import time

t0 = time.time()

status = st.status("Démarrage de l'analyse…", expanded=True)

status.write("1/6 Lecture du PDF + extraction texte (OCR si besoin)…")

# Crédit consommé ✅
# On copie le fichier uploadé en mémoire pour pouvoir le relire plusieurs fois (seek/open).
file_obj = io.BytesIO(uploaded.getvalue())
text, used_ocr, page_images, page_texts, page_ocr_flags = extract_text_auto_per_page(file_obj, dpi=DPI, force_ocr=OCR_FORCE)

status.write(f"✅ Texte extrait (OCR utilisé: {used_ocr})")

status.write("2/6 Vérification du document…")
ok_doc, msg_doc, doc_dbg = validate_uploaded_pdf(page_texts)
if not ok_doc:
    status.update(label="Analyse interrompue", state="error")
    st.error(msg_doc)
    if DEBUG:
        st.json(doc_dbg)
    st.stop()

fmt, fmt_dbg = detect_format(text)

status.write(f"✅ Document valide — format détecté: {fmt}")

status.write("3/6 Extraction des champs principaux…")

if DEBUG:
    st.write(f"Format détecté : **{fmt}**")
    st.json({"ocr": used_ocr, **fmt_dbg})
    wit
