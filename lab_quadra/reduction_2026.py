"""Diagnostic prudent de ventilation RGDU 2026, indépendant des codes Quadra.

Un montant réglementaire estimé ne vaut jamais validation d'un montant à
payer à un organisme. Les taux particuliers non démontrables ne sont pas
extrapolés depuis une autre entreprise.
"""
from decimal import Decimal, ROUND_HALF_UP
import re
import unicodedata

REGULAR_RATE = Decimal("6.01")
STANDARD_DENOMINATOR_LT50 = Decimal("39.81")


def normalize(value):
    t = unicodedata.normalize("NFKD", str(value))
    return re.sub(r"\s+", " ", "".join(
        c for c in t if not unicodedata.combining(c)).upper()).strip()


def observed_t1_rate(lines):
    """Taux patronal de retraite T1 calculé sur assiette et montant imprimés.

    La rubrique Q/E n'est PAS un indicateur de taux ni de collecteur.
    Plusieurs taux contradictoires interdisent la conclusion.
    """
    candidates = []
    for line in lines:
        label = normalize(line.get("label", ""))
        if not re.search(r"COMPLEMENTAIRE TRANCHE 1\b", label):
            continue
        base, pat = line.get("base"), line.get("pat")
        if not isinstance(base, int) or not isinstance(pat, int):
            continue
        if base <= 0 or pat <= 0:
            continue
        candidates.append(Decimal(pat) * 100 / Decimal(base))
    if not candidates or max(candidates)-min(candidates)>Decimal("0.025"):
        return None
    mean = sum(candidates) / Decimal(len(candidates))
    return mean.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def estimate_standard_split(reduction_cents, pension_t1_rate, fnal_rate=None):
    """Estimation informative seulement, paramètres de droit commun vérifiés.

    Si le FNAL ou le taux de retraite ne sont pas fiables, renvoie None.
    Notamment, ne pas appliquer 6,01/39,81 à une retraite dérogatoire 5,23.
    """
    if not isinstance(reduction_cents, int) or reduction_cents <= 0:
        return None
    if pension_t1_rate is None or abs(pension_t1_rate-REGULAR_RATE)>Decimal("0.02"):
        return None
    if fnal_rate is None:
        return None
    fnal = Decimal(str(fnal_rate))
    if fnal not in (Decimal("0.10"), Decimal("0.50")):
        return None
    denom = STANDARD_DENOMINATOR_LT50 + (fnal-Decimal("0.10"))
    part = Decimal(reduction_cents)*REGULAR_RATE/denom
    agirc = int(part.quantize(Decimal("1"),rounding=ROUND_HALF_UP))
    return {"urssaf": reduction_cents-agirc, "retraite": agirc,
            "status":"estimation_reglementaire", "denominator": denom}
