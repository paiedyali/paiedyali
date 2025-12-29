# appv3.py — QUADRA + SILAE (OCR auto), PDF synthèse téléchargeable
# Modifié pour workflow :
# 1) upload -> analyse partielle (validation lisibilité)
# 2) si lisible -> paiement (Stripe) demandé
# 3) après paiement (session_id), l'utilisateur lance l'analyse complète -> synthèse PDF
#
# Les fonctions existantes sont conservées ; uniquement la logique principale (UI) a été refactorée.
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
st.write("Tu déposes ton bulletin PDF → vérification rapide (gratuit) → paiement → analyse complète + export PDF.")

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

# ------------------------------------------------------------
# Credit DB (consommation 1 paiement = 1 analyse)
# ------------------------------------------------------------
import sqlite3
from contextlib import closing

def _db_path() -> str:
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

# ------------------------------------------------------------
# Options d'analyse (après paiement)
# ------------------------------------------------------------
OCR_FORCE = st.checkbox("Forcer l'OCR (si PDF image)", value=False)
DPI = st.slider("Qualité OCR (DPI) (pour analyse complète)", 150, 350, 250, 50)
uploaded = st.file_uploader("Dépose ton bulletin de salaire (PDF)", type=["pdf"])

# ------------------------------------------------------------
# (les fonctions utilitaires et d'extraction sont inchangées)
# ------------------------------------------------------------
# ... (Toutes les fonctions définies dans ton fichier d'origine sont conservées ci-dessous sans modification)
# Pour garder la réponse lisible, je ne répète pas ici toutes les fonctions déjà présentes (pdf_to_page_images,
# ocr_pdf_to_text, extract_text_auto, extract_text_auto_per_page, validate_uploaded_pdf, detect_format,
# extract_* etc.). Elles doivent rester exactement comme dans ton fichier d'origine.
#
# NOTE: dans l'implémentation finale (fichier complet), ces définitions existent inchangées au-dessus de la section MAIN.
#
# ------------------------------------------------------------
# MAIN (nouveau flux : partial -> paiement -> full)
# ------------------------------------------------------------
def run_full_analysis(file_obj: io.BytesIO, dpi_full: int, used_ocr_partial: bool = False, partial_text: str | None = None, page_images=None, page_texts=None, page_ocr_flags=None):
    """
    Fonction qui exécute l'analyse complète (coûteuse) — reprend la logique existante après consommation du crédit.
    Retourne (pdf_buf, fields, comments, debug_info)
    """
    import time
    t0 = time.time()
    status = st.status("Démarrage de l'analyse complète…", expanded=True)
    status.write("1/6 Lecture du PDF + extraction texte (OCR si besoin)…")

    # on relit le fichier
    file_obj.seek(0)
    # si on a une extraction partielle déjà, on peut réutiliser
    if partial_text is None or page_texts is None:
        text, used_ocr, page_images2, page_texts2, page_ocr_flags2 = extract_text_auto_per_page(file_obj, dpi=dpi_full, force_ocr=OCR_FORCE)
        page_images = page_images2
        page_texts = page_texts2
        page_ocr_flags = page_ocr_flags2
        used_ocr = used_ocr
    else:
        text = partial_text
        used_ocr = used_ocr_partial

    status.write(f"✅ Texte extrait (OCR utilisé: {used_ocr})")

    status.write("2/6 Vérification du document…")
    ok_doc, msg_doc, doc_dbg = validate_uploaded_pdf(page_texts)
    if not ok_doc:
        status.update(label="Analyse interrompue", state="error")
        st.error(msg_doc)
        if DEBUG:
            st.json(doc_dbg)
        return None, None, None, {"error": "doc_invalid", "doc_dbg": doc_dbg}

    fmt, fmt_dbg = detect_format(text)
    status.write(f"✅ Document valide — format détecté: {fmt}")

    status.write("3/6 Extraction des champs principaux…")

    if DEBUG:
        st.write(f"Format détecté : **{fmt}**")
        st.json({"ocr": used_ocr, **fmt_dbg})
        with st.expander("Texte extrait (début)"):
            st.text((text or "")[:12000])

    # Variables communes
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

    if fmt == "QUADRA":
        csg_nd, csg_nd_line = extract_csg_non_deductible_total(text)
    else:
        csg_nd, csg_nd_line = find_last_line_with_amount(
            text,
            include_patterns=[r"csg.*non\s+d[ée]duct", r"csg\/crds.*non\s+d[ée]duct", r"non\s+d[ée]duct.*imp[oô]t"],
            exclude_patterns=[],
        )

    acompte, acompte_line = extract_acompte(text, net_paye=net_paye, brut=brut)
    net_reference = round(net_paye + (acompte or 0.0), 2) if net_paye is not None else None

    charges_sal = None
    charges_pat = None
    charges_line = None
    charges_method = None
    cout_total = None
    cout_total_line = None
    cp = {"cp_n1": None, "cp_n": None, "cp_total": None}

    status.write("4/6 Extraction spécifique au format (QUADRA / SILAE)…")

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

    organismes_total = (
        round((charges_sal or 0.0) + (charges_pat or 0.0) + (csg_nd or 0.0), 2)
        if (charges_sal is not None or charges_pat is not None or csg_nd is not None)
        else None
    )

    status.write("✅ Extraction terminée")
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

    # PDF export
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

    status.write("6/6 Génération du PDF de synthèse…")
    status.update(label=f"Analyse terminée en {time.time()-t0:.1f}s", state="complete")

    st.download_button(
        "⬇️ Télécharger la synthèse PDF",
        data=pdf_buf.getvalue(),
        file_name="synthese_bulletin_particulier.pdf",
        mime="application/pdf",
    )

    # Debug
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

    return pdf_buf, fields, comments, {"time": time.time() - t0}


# MAIN UI flow
if uploaded is not None:
    st.success("PDF reçu ✅")

    # Analyse partielle (rapide) — objectif : vérifier lisibilité et format avant paiement
    st.markdown("### 1) Vérification rapide du PDF (gratuit)")
    partial_dpi = min(200, DPI)  # DPI plus bas pour la partie rapide
    file_obj = io.BytesIO(uploaded.getvalue())

    with st.spinner("Analyse rapide en cours (quelques secondes)…"):
        try:
            text_partial, used_ocr_partial, page_images, page_texts, page_ocr_flags = extract_text_auto_per_page(
                file_obj, dpi=partial_dpi, force_ocr=OCR_FORCE
            )
        except Exception as e:
            st.error("Erreur pendant l'analyse rapide (OCR). Vérifie le PDF ou active le mode debug.")
            if DEBUG:
                st.exception(e)
            st.stop()

    st.write(f"OCR utilisé : {used_ocr_partial}")
    ok_doc, msg_doc, doc_dbg = validate_uploaded_pdf(page_texts)
    if not ok_doc:
        st.error(msg_doc)
        if DEBUG:
            st.json(doc_dbg)
            with st.expander("Aperçu texte (partiel)"):
                st.text((text_partial or "")[:5000])
        st.stop()

    fmt, fmt_dbg = detect_format(text_partial)
    st.success(f"Document lisible et de type détecté : {fmt}")

    # Afficher aperçu et conseils si nécessaire
    with st.expander("Aperçu (début du texte extrait)"):
        st.text((text_partial or "")[:4000])

    # Paiement
    st.markdown("### 2) Paiement pour lancer l'analyse complète")
    st.write(f"Coût : {PRICE_EUR:.2f} € (paiement unique, 1 paiement = 1 analyse complète).")
    if PAYMENT_LINK:
        st.link_button("Payer maintenant", PAYMENT_LINK, type="primary")
        st.caption("Après paiement, tu seras redirigé ici (URL de succès Stripe doit contenir `session_id`).")
    else:
        st.error("Paiement non configuré : variable d'environnement STRIPE_PAYMENT_LINK manquante.")
        if ALLOW_NO_PAYMENT:
            st.info("Mode développement : paiement bypass activé.")
    if DEBUG:
        st.info("Mode debug : tu peux utiliser 'Bypass paiement' pour tester l'analyse complète sans paiement.")
        if st.button("Bypass paiement (debug)"):
            st.session_state._bypass_payment = True

    # Si l'utilisateur est revenu depuis Stripe avec session_id, on propose d'exécuter l'analyse complète
    session_id = _get_query_param("session_id") or _get_query_param("checkout_session_id")
    payment_ok, payment_reason = is_payment_ok()

    show_full_button = False
    if ALLOW_NO_PAYMENT or st.session_state.get("_bypass_payment"):
        show_full_button = True
    elif session_id:
        if payment_ok:
            # not yet consumed?
            if credit_is_consumed(session_id):
                st.error("🔒 Ce paiement a déjà été utilisé : 1 paiement = 1 analyse.")
            else:
                show_full_button = True
        else:
            st.warning(f"Paiement non confirmé : {payment_reason}. Si tu viens du lien de paiement, attends la redirection ou vérifie la session.")
            if DEBUG:
                st.info("Debug : tu peux forcer le démarrage complet en cliquant sur 'Forcer analyse complète (debug)'.")
                if st.button("Forcer analyse complète (debug)"):
                    st.session_state._bypass_payment = True
                    show_full_button = True

    if show_full_button:
        if st.button("Lancer l'analyse complète (après paiement)", type="primary"):
            # Consommation du crédit (si applicable)
            if not (ALLOW_NO_PAYMENT or st.session_state.get("_bypass_payment")):
                if not session_id:
                    st.error("session_id manquant dans l'URL. Reviens depuis la page de succès Stripe.")
                    st.stop()
                if credit_is_consumed(session_id):
                    st.error("🔒 Ce paiement a déjà été utilisé : 1 paiement = 1 analyse.")
                    st.stop()
                if not credit_consume(session_id):
                    st.error("🔒 Impossible de consommer le crédit (déjà consommé).")
                    st.stop()
                # petite mémoire UI
                st.session_state.analysis_credit_used_for = session_id
            else:
                # mode dev bypass : on n'utilise pas la DB
                pass

            # lancer l'analyse complète (réutilise l'extraction partielle si disponible)
            pdf_buf, fields, comments, dbg = run_full_analysis(
                io.BytesIO(uploaded.getvalue()), dpi_full=DPI, used_ocr_partial=used_ocr_partial, partial_text=text_partial, page_images=page_images, page_texts=page_texts, page_ocr_flags=page_ocr_flags
            )

            # run_full_analysis gère l'affichage et le téléchargement
    else:
        st.info("Pour lancer l'analyse complète, effectue le paiement via le bouton ci‑dessus. Une fois redirigé ici avec `session_id`, clique sur 'Lancer l'analyse complète'.")
        if DEBUG:
            st.write(f"[debug] payment_ok={payment_ok}, reason={payment_reason}, session_id={session_id}")

else:
    st.info("En attente d'un PDF…")
