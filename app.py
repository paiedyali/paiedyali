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
# Helper : extraction texte + OCR automatique
# ------------------------------------------------------------
def pdf_to_page_images(file, dpi=250):
    """Convertit le PDF en images par page pour OCR."""
    file.seek(0)
    images = []
    with pdfplumber.open(file) as pdf:
        for p in pdf.pages:
            images.append(p.to_image(resolution=dpi).original)
    return images

def ocr_pdf_to_text(file, dpi=250, lang="fra") -> str:
    """OCR d'un PDF (scan ou image)."""
    imgs = pdf_to_page_images(file, dpi=dpi)
    out = []
    for im in imgs:
        out.append(pytesseract.image_to_string(im, lang=lang))
    return "\n".join(out)

def extract_text_auto(file, dpi=250, force_ocr=False):
    """Extraction classique + OCR si nécessaire."""
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
    """Extraction par page, avec OCR si nécessaire."""
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

    # OCR page par page si nécessaire
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
# Extraction des montants / valeurs
# ------------------------------------------------------------
def to_float_fr(s: str):
    """Convertir un montant au format français en float."""
    if s is None:
        return None
    s = s.replace("\xa0", " ").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def extract_amounts_money(line: str):
    """Extraire les montants (valeurs numériques) d'une ligne de texte."""
    raw = re.findall(r"\d[\d\s]*[.,]\d{2}", line)
    vals = []
    for r in raw:
        v = to_float_fr(r)
        if v is not None:
            vals.append(v)
    return vals

def extract_last_amount_money(line: str):
    """Extraire le dernier montant trouvé sur une ligne."""
    vals = extract_amounts_money(line)
    return vals[-1] if vals else None

def find_last_line_with_amount(text: str, include_patterns, exclude_patterns=None):
    """Trouver la dernière ligne contenant un montant et qui correspond à un pattern donné."""
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
    Stratégie : repérer un taux plausible puis prendre le montant qui suit immédiatement ce taux.
    Retour : (total, lines_used)
    """
    total = 0.0
    lines_used = []

    for line in (text or "").splitlines():
        low = line.lower()
        if "csg" not in low:
            continue
        if ("non déduct" not in low) and ("non deduct" not in low):
            continue

        toks = re.findall(r"\d+(?:[.,]\d+)?", line)
        if not toks:
            continue

        nums = []
        for t in toks:
            v = to_float_fr(t.replace(",", "."))
            if v is not None:
                nums.append(v)

        if len(nums) < 2:
            continue

        picked = None
        for i, v in enumerate(nums):
            if 0.1 <= v <= 20.0:  # taux plausible
                if i + 1 < len(nums):
                    cand = nums[i + 1]
                    if 0 <= cand <= 2000:
                        picked = cand
                        break

        if picked is None:
            small = [v for v in nums if 0 <= v <= 2000]
            if small:
                picked = min(small)

        if picked is not None:
            total += picked
            lines_used.append(line)

    if not lines_used:
        return None, None

    return round(total, 2), lines_used

def extract_acompte(text: str, net_paye: float | None = None, brut: float | None = None):
    """Extraction de l'acompte (avance, déjà versé)"""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    best = None
    best_line = None

    if net_paye is not None and net_paye > 0:
        max_ok = net_paye * 1.5
    elif brut is not None and brut > 0:
        max_ok = brut * 1.5
    else:
        max_ok = 6000

    def sanitize_year_glue(s: str) -> str:
        s2 = re.sub(r"\b20\d{2}\b", " ", s)
        s2 = re.sub(r"\b2\s+0\d{2}\b", " ", s2)
        return s2

    def pick_plausible(vals, has_year: bool):
        out = []
        for v in vals:
            if v is None or v <= 0:
                continue
            if v > max_ok:
                continue
            if has_year and v > 2000:
                continue
            out.append(v)
        if not out:
            return None
        return max(out)

    best = None
    best_line = None

    for line in lines:
        low = line.lower()
        if not (("acompte" in low) or ("avance" in low) or ("déjà vers" in low) or ("deja vers" in low)):
            continue

        has_year = bool(re.search(r"\b20\d{2}\b", line)) or bool(re.search(r"\b2\s+0\d{2}\b", line))

        # 1) extraction brute
        vals1 = extract_amounts_money(line)
        v = pick_plausible(vals1, has_year=has_year)

        # 2) fallback : on retire l'année si elle a été collée / bruitée
        if v is None and has_year:
            line2 = sanitize_year_glue(line)
            vals2 = extract_amounts_money(line2)
            v = pick_plausible(vals2, has_year=False)

        if v is not None:
            best = v
            best_line = line

    return best, best_line
# ------------------------------------------------------------
# PDF : Génération du résumé
# ------------------------------------------------------------
def wrap_title_for_box(title: str, max_chars=18):
    """Réduit le titre à un nombre de caractères max, avec un retour à la ligne si nécessaire."""
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
    """Affiche un carré avec un titre et une valeur."""
    c.setStrokeColor(colors.HexColor("#222222"))
    c.setFillColor(colors.HexColor(fill_hex))
    c.roundRect(x, y - h, w, h, 10, stroke=1, fill=1)

    title_lines = wrap_title_for_box(title, max_chars=18)
    c.setFillColor(colors.HexColor("#111111"))
    c.setFont("Helvetica-Bold", 9 if len(title_lines) == 2 else 10)
    ty = y - 16
    for tl in title_lines:
        c.drawString(x + 10, ty, tl)
        ty -= 11

    c.setFont("Helvetica-Bold", 13)
    c.drawCentredString(x + w / 2, y - h + 22, value)

def build_pdf(fields, comments, fmt_name):
    """Génère le PDF du résumé de salaire."""
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

    # Ajout des autres champs : brut, net, charges, acompte, etc.
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

    # Affichage de la synthèse
    y_after_path = step_y - step_h - 18
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

    # Génération du PDF avec les commentaires
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
        yy -= 6

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

# ------------------------------------------------------------
# MODE FINAL : après paiement, récupérer le PDF depuis R2 (sans re-upload)
# ------------------------------------------------------------
stored_pdf_bytes = None
if MODE != "precheck" and paid_ok and client_ref:
    stored_pdf_bytes, r2_reason = r2_get_pdf(client_ref)
    if stored_pdf_bytes is None:
        st.warning(f"Paiement OK mais fichier introuvable dans R2 ({r2_reason}).")
        st.info("➡️ Fallback : re-uploade ton PDF ci-dessous.")

# ------------------------------------------------------------
# Affichage UI : on autorise l'analyse si
# - soit on a le PDF stocké en R2
# - soit l'utilisateur a uploadé un PDF
# ------------------------------------------------------------
has_pdf = (stored_pdf_bytes is not None) or (uploaded is not None)

if not has_pdf:
    st.info("En attente d'un PDF…")
    st.stop()

st.success("PDF reçu ✅")

if st.button("Analyser le bulletin", type="primary"):
    import time
    t0 = time.time()

    status = st.status("Démarrage de l'analyse…", expanded=True)
    status.write("1/6 Lecture du PDF + extraction texte (OCR si besoin)…")

    # ✅ Choix de la source du PDF (DANS le bouton)
    if stored_pdf_bytes is not None:
        file_obj = io.BytesIO(stored_pdf_bytes)
    else:
        file_obj = io.BytesIO(uploaded.getvalue())

    # ✅ Extraction
    text, used_ocr, page_images, page_texts, page_ocr_flags = extract_text_auto_per_page(
        file_obj, dpi=DPI, force_ocr=OCR_FORCE
    )
    status.write(f"✅ Texte extrait (OCR utilisé: {used_ocr})")

    # ✅ Vérif bulletin
    status.write("2/6 Vérification du document…")
    ok_doc, msg_doc, doc_dbg = validate_uploaded_pdf(page_texts)
    if not ok_doc:
        status.update(label="Analyse interrompue", state="error")
        st.error(msg_doc)
        if DEBUG:
            st.json(doc_dbg)
        st.stop()

    # ✅ Format
    fmt, fmt_dbg = detect_format(text)
    status.write(f"✅ Document valide — format détecté: {fmt}")
    status.write("3/6 Extraction des champs principaux…")

    # DEBUG: uniquement affichage
    if DEBUG:
        st.write(f"Format détecté : **{fmt}**")
        st.json({"ocr": used_ocr, **fmt_dbg})
        with st.expander("Texte extrait (début)") :
            st.text((text or "")[:12000])

    # Récupération des autres champs (extraction)
    period, period_line = extract_period(text)
    brut, brut_line = find_last_line_with_amount(
        text,
        include_patterns=[r"salaire\s+brut", r"\bbrut\b"],
        exclude_patterns=[r"net", r"imposable", r"csg", r"crds"],
    )

    net_paye, net_paye_line = find_last_line_with_amount(
        text,
        include_patterns=[r"net\s+paye", r"net\s+payé", r"net\s+à\s+payer", r"net\s+a\s+payer"],
        exclude_patterns=[r"avant\s+imp", r"imposable"],
    )

    pas, pas_line = find_last_line_with_amount(
        text,
        include_patterns=[r"imp[oô]t\s+sur\s+le\s+revenu", r"pr[ée]l[èe]vement\s+à\s+la\s+source", r"\bpas\b"],
        exclude_patterns=[r"csg", r"crds", r"deduct", r"non\s+deduct"],
    )

    # Extraction des informations selon le format
    # et génération de la synthèse avec les commentaires
