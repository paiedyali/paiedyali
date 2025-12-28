import io
import re
import datetime as dt
import os
import uuid
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import streamlit as st
import pdfplumber
import pytesseract
from pytesseract import Output
import boto3

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.pdfgen import canvas

# ------------------------------------------------------------
# Config paiement (doit être avant is_payment_ok)
# ------------------------------------------------------------
PRICE_EUR = 7.50
PAYMENT_LINK = os.getenv("STRIPE_PAYMENT_LINK", "").strip()
ALLOW_NO_PAYMENT = os.getenv("ALLOW_NO_PAYMENT", "false").lower() == "true"

# Debug technique (sans UI) : utile si ton app plante avant d'afficher le checkbox
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ------------------------------------------------------------
# Query params + mode
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

MODE = (_get_query_param("mode") or "final").lower()  # precheck | final

# ------------------------------------------------------------
# Stripe check
# ------------------------------------------------------------
def is_payment_ok() -> tuple[bool, str, str | None]:
    if ALLOW_NO_PAYMENT:
        return True, "bypass", None

    session_id = _get_query_param("session_id") or _get_query_param("checkout_session_id")
    if not session_id:
        return False, "missing_session_id", None

    secret_key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not secret_key:
        return False, "missing_STRIPE_SECRET_KEY", None

    try:
        import stripe  # lazy import
        stripe.api_key = secret_key
        s = stripe.checkout.Session.retrieve(session_id)

        paid = (getattr(s, "payment_status", None) == "paid") and (
            getattr(s, "status", None) in ("complete", "paid", None)
        )
        client_ref = getattr(s, "client_reference_id", None)
        return bool(paid), ("paid" if paid else "not_paid"), client_ref
    except Exception as e:
        return False, f"stripe_error:{type(e).__name__}", None

# IMPORTANT : on calcule paid_ok AVANT de l'utiliser
paid_ok, paid_reason, client_ref = is_payment_ok()

# ------------------------------------------------------------
# Paywall (uniquement en FINAL)
# ------------------------------------------------------------
if (MODE != "precheck") and (not paid_ok):
    st.markdown("## Vérification — 7,50 €")
    st.write("Pour analyser votre bulletin, une vérification coûte **7,50 €** (paiement unique).")
    if PAYMENT_LINK:
        st.link_button("Payer 7,50 €", PAYMENT_LINK, type="primary")
        st.caption("Après paiement, vous serez redirigé ici automatiquement.")
    else:
        st.error("Paiement non configuré : variable d'environnement STRIPE_PAYMENT_LINK manquante.")

    # Debug "early" (avant le checkbox UI)
    if DEBUG:
        st.info(f"[debug] accès refusé: {paid_reason}")

    st.stop()
# ------------------------------------------------------------
# Stockage Cloudflare R2 (S3-compatible) — PDF temporaire
# ------------------------------------------------------------
def _r2_client():
    endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
    key_id = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()

    if not endpoint or not key_id or not secret:
        return None, "missing_R2_env"

    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name="auto",
        )
        return s3, "ok"
    except Exception as e:
        return None, f"r2_client_error:{type(e).__name__}"

def _r2_bucket():
    return os.getenv("R2_BUCKET", "").strip()

def r2_put_pdf(pdf_bytes: bytes, precheck_id: str) -> tuple[bool, str]:
    s3, reason = _r2_client()
    if not s3:
        return False, reason

    bucket = _r2_bucket()
    if not bucket:
        return False, "missing_R2_BUCKET"

    key = f"prechecks/{precheck_id}.pdf"

    try:
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=pdf_bytes,
            ContentType="application/pdf",
        )
        return True, key
    except Exception as e:
        return False, f"r2_put_error:{type(e).__name__}"

def r2_get_pdf(precheck_id: str) -> tuple[bytes | None, str]:
    s3, reason = _r2_client()
    if not s3:
        return None, reason

    bucket = _r2_bucket()
    if not bucket:
        return None, "missing_R2_BUCKET"

    key = f"prechecks/{precheck_id}.pdf"

    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read(), "ok"
    except Exception as e:
        return None, f"r2_get_error:{type(e).__name__}"

# ------------------------------------------------------------
# Helper URL : ajout de paramètres (Stripe client_reference_id)
# ------------------------------------------------------------
def add_query_params(url: str, params: dict) -> str:
    u = urlparse(url)
    q = dict(parse_qsl(u.query, keep_blank_values=True))
    q.update({k: str(v) for k, v in params.items() if v is not None})
    return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(q), u.fragment))
# ------------------------------------------------------------
# Helper de nettoyage
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


def norm_spaces(s: str) -> str:
    s = (s or "").replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()


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
    t = text or ""

    def repl_date(m):
        a = dedouble_digits_if_all_pairs(m.group(1))
        b = dedouble_digits_if_all_pairs(m.group(2))
        c = dedouble_digits_if_all_pairs(m.group(3))
        return f"{a}/{b}/{c}"

    t = re.sub(r"\b(\d{2,4})[\/\-]{1,2}(\d{2,4})[\/\-]{1,2}(\d{4,8})\b", repl_date, t)
    t = re.sub(r"\b\d{8}\b", lambda m: dedouble_digits_if_all_pairs(m.group(0)), t)
    return t


# ------------------------------------------------------------
# Fonction d'extraction des données du PDF (OCR si nécessaire)
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
        for p in pdf.pages:
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
# ------------------------------------------------------------
# Fonctions d'extraction des montants (Net, Brut, Cotisations, etc.)
# ------------------------------------------------------------
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
# UI upload
# ------------------------------------------------------------

OCR_FORCE = st.checkbox("Forcer l'OCR (si PDF image)", value=False)
DPI = st.slider("Qualité OCR (DPI)", 150, 350, 250, 50)
uploaded = st.file_uploader("Dépose ton bulletin de salaire (PDF)", type=["pdf"])

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
    """Retourne une clé canonique (année, mois) pour comparer des périodes de bulletin."""
    if not period_str:
        return None
    s = norm_spaces(period_str).lower()

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
    m = re.search(
        r"\b(\d{2})[\/\-](\d{2})[\/\-](\d{4})\b.*?\b(\d{2})[\/\-](\d{2})[\/\-](\d{4})\b",
        s
    )
    if m:
        m1, y1 = int(m.group(2)), int(m.group(3))
        m2, y2 = int(m.group(5)), int(m.group(6))
        if (m1, y1) == (m2, y2):
            return (y1, m1)
        return (y1, m1, y2, m2)

    # cas: "11/2025" ou "11-2025"
    m = re.search(r"\b(0?[1-9]|1[0-2])[\/\-](20\d{2})\b", s)
    if m:
        return (int(m.group(2)), int(m.group(1)))

    # cas: "novembre 2025"
    for name, num in mois.items():
        if re.search(rf"\b{name}\b", s):
            my = re.search(r"\b(20\d{2})\b", s)
            if my:
                return (int(my.group(1)), num)

    return None
