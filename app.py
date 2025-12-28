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
# Configuration du paiement Stripe
# ------------------------------------------------------------
PRICE_EUR = 7.50
PAYMENT_LINK = os.getenv("STRIPE_PAYMENT_LINK", "").strip()  # lien Stripe pour le paiement
ALLOW_NO_PAYMENT = os.getenv("ALLOW_NO_PAYMENT", "false").lower() == "true"  # Si on permet les analyses sans paiement

# Mode debug
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ------------------------------------------------------------
# Query params + mode
# ------------------------------------------------------------
def _get_query_param(name: str):
    """Compatibilité Streamlit: query_params ou experimental_get_query_params."""
    if hasattr(st, "query_params"):
        return st.query_params.get(name)
    qp = st.experimental_get_query_params()
    v = qp.get(name)
    if isinstance(v, list):
        return v[0] if v else None
    return v

MODE = (_get_query_param("mode") or "final").lower()  # precheck | final

# ------------------------------------------------------------
# Stripe check pour vérifier le paiement
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
# Validation du document PDF (fonction principale)
# ------------------------------------------------------------
def validate_uploaded_pdf(page_texts: list[str]) -> tuple[bool, str, dict]:
    n = len(page_texts or [])
    all_text = "\n".join(page_texts or [])

    ok_ps, dbg_ps = is_likely_payslip(all_text)
    fmt, fmt_dbg = detect_format(all_text)

    if (not ok_ps) or (fmt == "INCONNU"):
        return (
            False,
            "🔒 Document refusé : ce PDF ne ressemble pas à un **bulletin de salaire** (ou format non reconnu).",
            {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n},
        )

    if n <= 1:
        return True, "OK", {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n}

    if n > 2:
        return (
            False,
            "🔒 Document refusé : PDF **multi-pages** (>2) non accepté. Dépose uniquement le bulletin concerné.",
            {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n},
        )

    p1, p2 = page_texts[0], page_texts[1]

    period_all, _ = extract_period(all_text)
    period_1, _ = extract_period(p1)
    period_2, _ = extract_period(p2)

    period_ref = period_1 or period_all
    key_ref = period_key(period_ref)
    key_p2 = period_key(period_2)

    if not key_ref:
        return (
            False,
            "🔒 Document refusé : bulletin sur 2 pages mais **période** illisible (je ne peux pas vérifier que c'est la même).",
            {
                "payslip_dbg": dbg_ps,
                "fmt": fmt,
                "fmt_dbg": fmt_dbg,
                "pages": n,
                "periods": [period_1, period_2, period_all],
                "period_keys": [key_ref, key_p2],
            },
        )

    if key_p2 and (key_p2 != key_ref):
        return (
            False,
            "🔒 Document refusé : bulletin sur 2 pages mais **période différente** entre les pages.",
            {
                "payslip_dbg": dbg_ps,
                "fmt": fmt,
                "fmt_dbg": fmt_dbg,
                "pages": n,
                "periods": [period_1, period_2, period_all],
                "period_keys": [key_ref, key_p2],
            },
        )

    emp_1 = extract_employee_id(p1) or extract_employee_id(all_text)
    emp_2 = extract_employee_id(p2) or extract_employee_id(all_text)

    if not emp_1:
        return (
            False,
            "🔒 Document refusé : bulletin sur 2 pages mais **salarié** illisible (je ne peux pas vérifier que c'est le même).",
            {"payslip_dbg": dbg_ps, "fmt": fmt, "fmt_dbg": fmt_dbg, "pages": n, "employee": [emp_1, emp_2]},
        )

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
def extract_period(text: str):
    """Extrait la période du bulletin de salaire (par exemple 'Du 01/01/2025 au 31/01/2025')."""
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

    return None, None

def extract_employee_id(text: str):
    """Extrait l'identité de l'employé du bulletin (par exemple 'DUPONT JEAN')."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    head = lines[:60]

    for l in head:
        m = re.search(r"(salari[ée]|salarie|employ[ée]|employe)\s*[:\-]\s*(.+)$", l, re.IGNORECASE)
        if m:
            name = _clean_person_token(m.group(2))
            if len(name) >= 5:
                return name.upper()

    return None
# ------------------------------------------------------------
# Extraction des montants (salaires, cotisations, etc.)
# ------------------------------------------------------------

def to_float_fr(s: str):
    """Convertir une chaîne en nombre flottant avec le format français."""
    if s is None:
        return None
    s = s.replace("\xa0", " ").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def extract_amounts_money(line: str):
    """Extrait les montants d'argent d'une ligne."""
    raw = re.findall(r"\d[\d\s]*[.,]\d{2}", line)
    vals = []
    for r in raw:
        v = to_float_fr(r)
        if v is not None:
            vals.append(v)
    return vals


def extract_last_amount_money(line: str):
    """Extrait le dernier montant d'argent d'une ligne."""
    vals = extract_amounts_money(line)
    return vals[-1] if vals else None


def find_last_line_with_amount(text: str, include_patterns, exclude_patterns=None):
    """Trouver la dernière ligne contenant un montant d'argent qui correspond aux patterns."""
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
    """Formater un montant en euros avec le format français."""
    if v is None:
        return "-"
    s = f"{v:,.2f}".replace(",", " ").replace(".", ",")
    return f"{s} EUR"
# ------------------------------------------------------------
# Extraction des informations spécifiques au format QUADRA et SILAE
# ------------------------------------------------------------

def extract_charges_quadra(text: str):
    """Extraction des charges salariales et patronales pour le format QUADRA."""
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
    """Extraction du total de la CSG non déductible pour QUADRA."""
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
    """Extraction de l'acompte pour QUADRA et SILAE."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    best = None
    best_line = None

    # Bornes plausibles (mensuel)
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
# ------------------------------------------------------------
# SILAE — coût global (mensuel) via OCR zone, CP via OCR bas gauche
# ------------------------------------------------------------

def extract_cout_global_fast(page_img):
    """Version rapide : OCR d'une zone en haut de page puis recherche 'coût global'."""
    try:
        W, H = page_img.width, page_img.height
        crop = page_img.crop((0, 0, W, int(H * 0.40)))
        zone_text = norm_spaces(pytesseract.image_to_string(crop, lang="fra"))
        lines = [l.strip() for l in zone_text.splitlines() if l.strip()]
        for i, l in enumerate(lines):
            ll = l.lower()
            if ("coût global" in ll) or ("cout global" in ll):
                chunk = " ".join(lines[i:i + 4])
                vals = extract_amounts_money(chunk)
                vals_ok = [v for v in vals if 0 < v < 20000]
                if vals_ok:
                    return vals_ok[0], f"fast_ocr: {chunk[:200]}"
        return None, f"fast_ocr_nohit: {zone_text[:220]}"
    except Exception as e:
        return None, f"fast_ocr_error:{type(e).__name__}"


def extract_cout_global_from_image(page_img):
    """Repère le titre 'Coût global' et lit le montant mensuel juste en dessous."""
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
    """Acompte / avance / déjà versé (QUADRA + SILAE)."""
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

    for line in lines:
        low = line.lower()
        if not (("acompte" in low) or ("avance" in low) or ("déjà vers" in low) or ("deja vers" in low)):
            continue

        has_year = bool(re.search(r"\b20\d{2}\b", line)) or bool(re.search(r"\b2\s+0\d{2}\b", line))

        vals1 = extract_amounts_money(line)
        v = pick_plausible(vals1, has_year=has_year)

        if v is None and has_year:
            line2 = sanitize_year_glue(line)
            vals2 = extract_amounts_money(line2)
            v = pick_plausible(vals2, has_year=False)

        if v is not None:
            best = v
            best_line = line

    return best, best_line
# ------------------------------------------------------------
# Génération du PDF de synthèse
# ------------------------------------------------------------
def build_comments(brut, charges_sal, csg_nd, pas, charges_pat, acompte):
    out = []

    if acompte is not None and acompte > 0:
        out.append("• Acompte détecté : le 'net payé' correspond à ce qui arrive ce mois-ci, pas forcément au net total du bulletin.")

    if brut is not None and brut > 0:
        prelev = (charges_sal or 0.0) + (csg_nd or 0.0)
        ratio = round((prelev / brut) * 100, 1)
        out.append(f"• Prélèvements sur ton brut : {ratio}% (cotisations salariales + CSG non déductible).")

    if pas is not None:
        if pas == 0:
            out.append("• Impôt à 0 EUR : aujourd'hui, le fisc observe. Sans bouger. Moment rare.")
        else:
            out.append("• Le PAS est prélevé sur ton salaire, puis versé au fisc par l'employeur.")

    if csg_nd is not None and csg_nd > 0:
        out.append("• La CSG non déductible : tu ne la touches pas, mais elle augmente ton net imposable. Plot twist fiscal.")

    if charges_pat is not None and charges_pat > 0:
        out.append("• Ton employeur dépense plus que ce que tu touches. Ce n'est pas toi, c'est le système.")

    return out[:7]


# ------------------------------------------------------------
# PDF export (résumé)
# ------------------------------------------------------------
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
# MAIN : Analyse du bulletin de salaire
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
# Affichage de l'UI : on autorise l'analyse si
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

    # ✅ Vérification du document
    status.write("2/6 Vérification du document…")
    ok_doc, msg_doc, doc_dbg = validate_uploaded_pdf(page_texts)
    if not ok_doc:
        status.update(label="Analyse interrompue", state="error")
        st.error(msg_doc)
        if DEBUG:
            st.json(doc_dbg)
        st.stop()

    # ✅ Format du bulletin
    fmt, fmt_dbg = detect_format(text)
    status.write(f"✅ Document valide — format détecté: {fmt}")
    status.write("3/6 Extraction des champs principaux…")

    # DEBUG: uniquement affichage
    if DEBUG:
        st.write(f"Format détecté : **{fmt}**")
        st.json({"ocr": used_ocr, **fmt_dbg})
        with st.expander("Texte extrait (début)"):
            st.text((text or "")[:12000])

    # ------------------------------------------------------------
    # Extraction commune des données (TOUJOURS dans le bouton)
    # ------------------------------------------------------------
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

    # CSG non déductible
    if fmt == "QUADRA":
        csg_nd, csg_nd_line = extract_csg_non_deductible_total(text)
    else:
        csg_nd, csg_nd_line = find_last_line_with_amount(
            text,
            include_patterns=[r"csg.*non\s+d[ée]duct", r"csg\/crds.*non\s+d[ée]duct", r"non\s+d[ée]duct.*imp[oô]t"],
            exclude_patterns=[],
        )

    # Acompte + net reconstitué
    acompte, acompte_line = extract_acompte(text, net_paye=net_paye, brut=brut)
    net_reference = round(net_paye + (acompte or 0.0), 2) if net_paye is not None else None

    # Init variables format
    charges_sal = None
    charges_pat = None
    charges_line = None
    charges_method = None
    cout_total = None
    cout_total_line = None
    cp = {"cp_n1": None, "cp_n": None, "cp_total": None}

    status.write("4/6 Extraction spécifique au format (QUADRA / SILAE)…")

    # ------------------------------------------------------------
    # Extraction spécifique au format QUADRA ou SILAE
    # ------------------------------------------------------------
    if fmt == "QUADRA":
        charges_sal, charges_pat, charges_line, charges_method = extract_charges_quadra(text)
        cout_total, cout_total_line = extract_total_verse_employeur_quadra(text, brut=brut, net_paye=net_paye)
        cp = extract_cp_quadra(text)

    elif fmt == "SILAE":
        lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
        low_lines = [l.lower() for l in lines]
        for i, ll in enumerate(low_lines):
            if ("total" in ll) and ("cotis" in ll) and ("contrib" in ll):
                vals = extract_amounts_money(lines[i])
                if len(vals) >= 2:
                    charges_sal, charges_pat = vals[0], vals[1]
                    charges_line = lines[i]
                    charges_method = "silae_total_cotis_contrib"
                    break

        if page_images:
            idx_page = _choose_silae_page_index(page_images, page_texts)
            page_img = page_images[idx_page]
            page_txt = page_texts[idx_page] if page_texts else None

            status.write(f"SILAE : page analysée = {idx_page + 1}/{len(page_images)}…")
            status.write("SILAE : extraction coût global + congés (optimisée)…")

            cout_total, cout_total_line, cp, silae_dbg = extract_silae_cost_and_cp(page_img, page_text=page_txt)

            if DEBUG:
                st.json({"silae_debug": silae_dbg, "cout_total_line": cout_total_line})

    # ------------------------------------------------------------
    # Totaux + synthèse
    # ------------------------------------------------------------
    organismes_total = (
        round((charges_sal or 0.0) + (charges_pat or 0.0) + (csg_nd or 0.0), 2)
        if (charges_sal is not None or charges_pat is not None or csg_nd is not None)
        else None
    )

    status.write("5/6 Affichage de la synthèse…")
    st.subheader("🎯 L'essentiel, sans jargon")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💸 Ce qui arrive sur ton compte")
        st.metric("Net payé (reçu)", eur(net_paye))
        if acompte and acompte > 0:
            st.metric("Acompte déjà versé", eur(acompte))
            st.metric("Net reconstitué", eur(net_reference))
        st.metric("Impôt (PAS)", eur(pas))

    with col2:
        st.markdown("### 💼 D'où ça part")
        st.metric("Brut", eur(brut))
        st.metric("Cotisations salariales", eur(charges_sal))
        st.metric("CSG non déductible", eur(csg_nd))

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 🏗️ Côté employeur")
        st.metric("Cotisations patronales", eur(charges_pat))
        st.metric("Organismes sociaux (total)", eur(organismes_total))
        st.metric("Coût total employeur (référence bulletin)", eur(cout_total))

    with col4:
        st.markdown("### 🌴 Congés disponibles (solde)")
        if cp.get("cp_total") is not None:
            st.metric("CP N-1 (solde)", f"{cp.get('cp_n1'):.2f} j" if cp.get("cp_n1") is not None else "-")
            st.metric("CP N (solde)", f"{cp.get('cp_n'):.2f} j" if cp.get("cp_n") is not None else "-")
            st.metric("Total", f"{cp.get('cp_total'):.2f} j")
        else:
            st.info("Congés : non lisibles automatiquement sur ce PDF.")

    st.subheader("😄 Ce qu'il faut retenir")
    comments = build_comments(brut, charges_sal, csg_nd, pas, charges_pat, acompte)
    for cmt in comments:
        st.write(cmt)

    # ------------------------------------------------------------
    # PDF export
    # ------------------------------------------------------------
    status.write("6/6 Génération du PDF de synthèse…")

    fields = {
        "period": period,
        "brut": brut,
        "net_paye": net_paye,
        "net_reference": net_reference,
        "acompte": acompte,
        "pas": pas,
        "charges_sal": charges_sal,
        "charges_pat": charges_pat,
        "csg_non_deductible": csg_nd,
        "organismes_total": organismes_total,
        "cout_total": cout_total,
        "cp": cp,
    }

    pdf_buf = build_pdf(fields, comments, fmt_name=fmt)

    status.update(label=f"Analyse terminée en {time.time() - t0:.1f}s", state="complete")

    st.download_button(
        "⬇️ Télécharger la synthèse PDF",
        data=pdf_buf.getvalue(),
        file_name="synthese_bulletin_particulier.pdf",
        mime="application/pdf",
    )

    # Debug complet (lignes sources)
    if DEBUG:
        st.subheader("🔧 Debug (lignes sources)")
        st.json(
            {
                "format": fmt,
                "ocr_used": used_ocr,
                "period_line": period_line,
                "brut_line": brut_line,
                "net_paye_line": net_paye_line,
                "pas_line": pas_line,
                "csg_nd_line": csg_nd_line,
                "acompte_line": acompte_line,
                "charges_line": charges_line,
                "charges_method": charges_method,
                "cout_total_line": cout_total_line,
                "cp": cp,
                "fmt_dbg": fmt_dbg,
            }
        )
# ------------------------------------------------------------
# FINAL : UI de fin et logique de paiement
# ------------------------------------------------------------

# Vérification de paiement et redirection en cas de succès
if MODE != "precheck" and paid_ok and client_ref:
    stored_pdf_bytes, r2_reason = r2_get_pdf(client_ref)
    if stored_pdf_bytes is None:
        st.warning(f"Paiement OK mais fichier introuvable dans R2 ({r2_reason}).")
        st.info("➡️ Fallback : re-uploade ton PDF ci-dessous.")
        
# Si le PDF est disponible, on permet l'analyse
has_pdf = (stored_pdf_bytes is not None) or (uploaded is not None)

if not has_pdf:
    st.info("En attente d'un PDF…")
    st.stop()

st.success("PDF reçu ✅")

# Analyse du bulletin lorsqu'on appuie sur le bouton
if st.button("Analyser le bulletin", type="primary"):
    import time
    t0 = time.time()

    # Démarrage de l'analyse
    status = st.status("Démarrage de l'analyse…", expanded=True)
    status.write("1/6 Lecture du PDF + extraction texte (OCR si besoin)…")

    if stored_pdf_bytes is not None:
        file_obj = io.BytesIO(stored_pdf_bytes)
    else:
        file_obj = io.BytesIO(uploaded.getvalue())

    # Extraction du texte avec ou sans OCR
    text, used_ocr, page_images, page_texts, page_ocr_flags = extract_text_auto_per_page(
        file_obj, dpi=DPI, force_ocr=OCR_FORCE
    )
    status.write(f"✅ Texte extrait (OCR utilisé: {used_ocr})")

    # Vérification de la validité du document
    status.write("2/6 Vérification du document…")
    ok_doc, msg_doc, doc_dbg = validate_uploaded_pdf(page_texts)
    if not ok_doc:
        status.update(label="Analyse interrompue", state="error")
        st.error(msg_doc)
        if DEBUG:
            st.json(doc_dbg)
        st.stop()

    # Détection du format
    fmt, fmt_dbg = detect_format(text)
    status.write(f"✅ Document valide — format détecté: {fmt}")
    status.write("3/6 Extraction des champs principaux…")

    # DEBUG : uniquement affichage
    if DEBUG:
        st.write(f"Format détecté : **{fmt}**")
        st.json({"ocr": used_ocr, **fmt_dbg})
        with st.expander("Texte extrait (début)"):
            st.text((text or "")[:12000])

    # Extraction des variables principales (brut, net, etc.)
    period, period_line = extract_period(text)
    brut, brut_line = find_last_line_with_amount(
        text, include_patterns=[r"salaire\s+brut", r"\bbrut\b"], exclude_patterns=[r"net", r"imposable", r"csg", r"crds"]
    )
    net_paye, net_paye_line = find_last_line_with_amount(
        text, include_patterns=[r"net\s+paye", r"net\s+payé", r"net\s+à\s+payer", r"net\s+a\s+payer"], exclude_patterns=[r"avant\s+imp", r"imposable"]
    )
    pas, pas_line = find_last_line_with_amount(
        text, include_patterns=[r"imp[oô]t\s+sur\s+le\s+revenu", r"pr[ée]l[èe]vement\s+à\s+la\s+source", r"\bpas\b"], exclude_patterns=[r"csg", r"crds", r"deduct", r"non\s+deduct"]
    )

    # CSG non déductible
    if fmt == "QUADRA":
        csg_nd, csg_nd_line = extract_csg_non_deductible_total(text)
    else:
        csg_nd, csg_nd_line = find_last_line_with_amount(
            text, include_patterns=[r"csg.*non\s+d[ée]duct", r"csg\/crds.*non\s+d[ée]duct", r"non\s+d[ée]duct.*imp[oô]t"], exclude_patterns=[]
        )

    # Acompte et net reconstitué
    acompte, acompte_line = extract_acompte(text, net_paye=net_paye, brut=brut)
    net_reference = round(net_paye + (acompte or 0.0), 2) if net_paye is not None else None

    # Initialisation des variables
    charges_sal = None
    charges_pat = None
    charges_line = None
    charges_method = None
    cout_total = None
    cout_total_line = None
    cp = {"cp_n1": None, "cp_n": None, "cp_total": None}

    status.write("4/6 Extraction spécifique au format (QUADRA / SILAE)…")

    # Extraction en fonction du format
    if fmt == "QUADRA":
        charges_sal, charges_pat, charges_line, charges_method = extract_charges_quadra(text)
        cout_total, cout_total_line = extract_total_verse_employeur_quadra(text, brut=brut, net_paye=net_paye)
        cp = extract_cp_quadra(text)

    elif fmt == "SILAE":
        lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
        low_lines = [l.lower() for l in lines]
        for i, ll in enumerate(low_lines):
            if ("total" in ll) and ("cotis" in ll) and ("contrib" in ll):
                vals = extract_amounts_money(lines[i])
                if len(vals) >= 2:
                    charges_sal, charges_pat = vals[0], vals[1]
                    charges_line = lines[i]
                    charges_method = "silae_total_cotis_contrib"
                    break

        if page_images:
            idx_page = _choose_silae_page_index(page_images, page_texts)
            page_img = page_images[idx_page]
            page_txt = page_texts[idx_page] if page_texts else None

            status.write(f"SILAE : page analysée = {idx_page + 1}/{len(page_images)}…")
            status.write("SILAE : extraction coût global + congés (optimisée)…")

            cout_total, cout_total_line, cp, silae_dbg = extract_silae_cost_and_cp(page_img, page_text=page_txt)

            if DEBUG:
                st.json({"silae_debug": silae_dbg, "cout_total_line": cout_total_line})

    # Calcul total des organismes sociaux
    organismes_total = (
        round((charges_sal or 0.0) + (charges_pat or 0.0) + (csg_nd or 0.0), 2)
        if (charges_sal is not None or charges_pat is not None or csg_nd is not None)
        else None
    )

    status.write("5/6 Affichage de la synthèse…")
    st.subheader("🎯 L'essentiel, sans jargon")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 💸 Ce qui arrive sur ton compte")
        st.metric("Net payé (reçu)", eur(net_paye))
        if acompte and acompte > 0:
            st.metric("Acompte déjà versé", eur(acompte))
            st.metric("Net reconstitué", eur(net_reference))
        st.metric("Impôt (PAS)", eur(pas))

    with col2:
        st.markdown("### 💼 D'où ça part")
        st.metric("Brut", eur(brut))
        st.metric("Cotisations salariales", eur(charges_sal))
        st.metric("CSG non déductible", eur(csg_nd))

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### 🏗️ Côté employeur")
        st.metric("Cotisations patronales", eur(charges_pat))
        st.metric("Organismes sociaux (total)", eur(organismes_total))
        st.metric("Coût total employeur (référence bulletin)", eur(cout_total))

    with col4:
        st.markdown("### 🌴 Congés disponibles (solde)")
        if cp.get("cp_total") is not None:
            st.metric("CP N-1 (solde)", f"{cp.get('cp_n1'):.2f} j" if cp.get("cp_n1") is not None else "-")
            st.metric("CP N (solde)", f"{cp.get('cp_n'):.2f} j" if cp.get("cp_n") is not None else "-")
            st.metric("Total", f"{cp.get('cp_total'):.2f} j")
        else:
            st.info("Congés : non lisibles automatiquement sur ce PDF.")

    st.subheader("😄 Ce qu'il faut retenir")
    comments = build_comments(brut, charges_sal, csg_nd, pas, charges_pat, acompte)
    for cmt in comments:
        st.write(cmt)

    # ------------------------------------------------------------
    # Génération du PDF de synthèse
    # ------------------------------------------------------------
    status.write("6/6 Génération du PDF de synthèse…")

    fields = {
        "period": period,
        "brut": brut,
        "net_paye": net_paye,
        "net_reference": net_reference,
        "acompte": acompte,
        "pas": pas,
        "charges_sal": charges_sal,
        "charges_pat": charges_pat,
        "csg_non_deductible": csg_nd,
        "organismes_total": organismes_total,
        "cout_total": cout_total,
        "cp": cp,
    }

    pdf_buf = build_pdf(fields, comments, fmt_name=fmt)

    status.update(label=f"Analyse terminée en {time.time() - t0:.1f}s", state="complete")

    st.download_button(
        "⬇️ Télécharger la synthèse PDF",
        data=pdf_buf.getvalue(),
        file_name="synthese_bulletin_particulier.pdf",
        mime="application/pdf",
    )

    # Debug complet (lignes sources)
    if DEBUG:
        st.subheader("🔧 Debug (lignes sources)")
        st.json(
            {
                "format": fmt,
                "ocr_used": used_ocr,
                "period_line": period_line,
                "brut_line": brut_line,
                "net_paye_line": net_paye_line,
                "pas_line": pas_line,
                "csg_nd_line": csg_nd_line,
                "acompte_line": acompte_line,
                "charges_line": charges_line,
                "charges_method": charges_method,
                "cout_total_line": cout_total_line,
                "cp": cp,
                "fmt_dbg": fmt_dbg,
            }
        )
