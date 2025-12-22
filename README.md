# paiedyali — Déploiement rapide (Streamlit + Docker)

## Lancer en local
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
```

## Activer le paiement (Stripe)
Ce projet attend un paramètre d'URL après paiement:
- `?session_id=cs_test_...`

Dans Stripe Checkout, configure la `success_url` vers:
- `https://VOTRE_DOMAINE_APP/?session_id={CHECKOUT_SESSION_ID}`

Variables d'environnement:
- `STRIPE_SECRET_KEY` : clé secrète Stripe (obligatoire pour vérifier la session)
- `STRIPE_PAYMENT_LINK` : lien de paiement (affiché si l'utilisateur n'a pas payé)
- `ALLOW_NO_PAYMENT=true` : bypass (dev/test)

## Déployer sur Render (Docker)
- Pousse ces fichiers sur GitHub
- Sur Render: "New" → "Blueprint" → sélectionne le repo (Render lit `render.yaml`)
- Renseigne les variables `STRIPE_SECRET_KEY` et `STRIPE_PAYMENT_LINK`

Fichiers:
- `app.py` : application Streamlit
- `Dockerfile` : image avec Tesseract (OCR)
- `requirements.txt`
- `render.yaml`
