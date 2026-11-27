import os,re,hmac,tempfile,subprocess,unicodedata
from pathlib import Path
from decimal import Decimal
from collections import defaultdict
import streamlit as st
from reduction_2026 import observed_t1_rate, estimate_standard_split
st.set_page_config(page_title="PaieDyali Entreprise - laboratoire Quadra",layout="wide")
st.title("PaieDyali Entreprise - Laboratoire Quadra")
st.caption("Réservé aux bulletins fictifs de test. Ventilations expérimentales, non certifiées.")
secret=os.getenv("PAIEDYALI_TEST_PASSWORD","")
if not secret:st.error("Mot de passe non configuré");st.stop()
pw=st.text_input("Mot de passe du laboratoire",type="password")
if not pw or not hmac.compare_digest(pw,secret):
    if pw:st.error("Accès refusé")
    st.stop()
MONEY=re.compile(r"-?\d+\.\d{2,4}")
# Codes de rubrique variables selon le dossier : E, EQ, EW, E_, SW.
# EMUT (mutuelle) et les rubriques apprenti doivent être acceptés.
# Le code ne détermine jamais l'organisme.
CODE=re.compile(r"^\s*((?:E[A-Z0-9_]{2,5}|SW[A-Z0-9_]{2,4}))\s+(.+)$")
def cents(s):return int((Decimal(s)*100).quantize(Decimal("1")))
def euro(c):return f"{c/100:,.2f} €".replace(","," ").replace(".",",")
def pct(part,base):
    if base is None or base<=0:return "—"
    return f"{part*100/base:.2f}".replace(".",",")+" %"

def extract_pas(line):
    """Return PAS withholding in cents, or None if the tax line has no amount."""
    m=re.search(r"(?i)(?:imp[oô]t sur le revenu pr[eé]lev[eé] [àa] la source|pr[eé]l[eè]vement [àa] la source)",line)
    if not m:return None
    nums=MONEY.findall(line[m.end():])
    # Columns: tax base, rate, withheld amount, optionally annual cumulative.
    if len(nums)<3:return None
    return cents(nums[2])
def infer_zero_pas(pas, tax_row_seen, net, net_before_tax, net_source):
    """PAS absent seulement si le net explicitement imprimé prouve 0 €."""
    return (pas is None and tax_row_seen and net_source=="bulletin"
            and net is not None and net_before_tax is not None
            and net==net_before_tax)

def extract_net_before_tax(line):
    """Montant explicitement imprime avant le PAS (aucune inference)."""
    m=re.search(r"(?i)NET\s+[AÀ]\s+PAYER\s+AVANT\s+IMP[OÔ]T\s+SUR\s+LE\s+REVENU",line)
    if not m:return None
    values=MONEY.findall(line[m.end():])
    # Montant sur la ligne de titre, avant la colonne laterale "Total".
    if not values:return None
    return cents(values[0])

def extract_net_paye(line):
    """Montant NET PAYE explicitement libelle, avec ou sans Euros."""
    m=re.search(r"(?i)\bNET\s+PAY[EÉ]\b(?!\s+AVANT)",line)
    if not m:return None
    after=line[m.end():]
    found=re.match(r"\s*:?\s*(-?\d+\.\d{2})(?:\s+Euros)?(?:\s|$)",after)
    return cents(found.group(1)) if found else None

def extract_net_salary_block(lines,index):
    """Valeur isolee du bloc 'Net a payer au salarie' des bulletins clarifies."""
    header=lines[index]
    m=re.search(r"(?i)Net\s+[àa]\s+payer\s+au\s+salari[eé]",header)
    if not m:return None
    same=MONEY.findall(header[m.end():])
    if len(same)==1:return cents(same[0])
    # Le montant est souvent sous deux lignes de convention collective.
    # Ne jamais lire au-dela des mentions legales ou de la page.
    for line in lines[index+1:index+18]:
        if re.search(r"(?i)A d[eé]faut de convention|Pour plus d.informations|Dans votre int[eé]r[eê]t",line):
            break
        found=re.fullmatch(r"\s*(-?\d+\.\d{2})\s*",line)
        if found:return cents(found.group(1))
    return None

def normalized_label(value):
    value=unicodedata.normalize("NFKD",value)
    value="".join(ch for ch in value if not unicodedata.combining(ch))
    value=value.upper().replace("’","'")
    return " ".join(re.sub(r"[^A-Z0-9]+"," ",value).split())

def non_contribution(label):
    # Le code Q de section ne suffit pas : des remboursements figurent
    # parfois sous Q801 à la suite des cotisations.
    v=normalized_label(label)
    # Sous-total documentaire déjà compris dans les cotisations détaillées.
    # Sur les bulletins longs avec sections Q, l'ancien moteur le prenait
    # pour une deuxième ligne de prévoyance (PREV dans le libellé).
    if re.search(r"^COTIS RETRAITE PREV F SANTE(?:$| )",v):
        return True
    return any(re.search(pattern,v) for pattern in (
        r"\bINDEMNITE\b.*\b(REPAS|PANIER|CASSE CROUTE|TRANSPORT|DEPLACEMENT|RESTAURANT|RESTO)\b",
        r"\bREMBOURSEMENT\b.*\bFRAIS\b",
        r"\bFRAIS\b.*\b(REPAS|PROFESSIONNELS|DEPLACEMENT)\b",
        r"\bPRIME\b.*\b(PANIER|SALISSURE)\b",
        # Retenues sur salaire apres cotisations : pas des charges sociales.
        r"\bSAISIE\b.*\b(ARRET|SUR SALAIRE|REMUNERATION)\b",
        r"\bACOMPTE\b",
        r"^PASS NAVIGO(?:$| )",
    ))

def health_supplement_is_mutuelle(label,raw_line,sal_amount,pat_amount):
    """Reconnaître une complémentaire santé partagée également.

    Sur AXE DV : 8 primes de 31,02 + 31,02; sur ELM la cotisation
    "Complémentaire Santé" est 11,52 + 17,28 et reste en prévoyance.
    Le code Q/E ne joue aucun rôle dans cette décision.
    """
    if normalized_label(label)!="COMPLEMENTAIRE SANTE":
        return False
    if sal_amount<=0 or pat_amount<=0:
        return False
    # Ne pas dépendre d'une colonne taux à position variable dans Quadra.
    # Une différence de 0,01 EUR est tolérée pour l'arrondi par moitié.
    return abs(sal_amount-pat_amount)<=1


def semantic_group(label,section=None):
    v=normalized_label(label)
    # Les codes E, EQ, EW, SW et Q n'identifient pas le collecteur.
    # Le texte et les regroupements sur le bulletin sont prioritaires.
    if non_contribution(label):return "EXCLUDE"
    # Quadra affiche une ligne informative sous le net, qui cite
    # "cotisations", "chomage" et "maladie" mais ne constitue pas une charge.
    # Ne jamais l'assimiler a une rubrique sociale.
    if v.startswith("DONT EVOLUTION DE LA REMUNERATION"):return "EXCLUDE"

    # La reduction generale publiee dans une ligne Q peut etre commune
    # URSSAF + AGIRC; une ligne detaillee doit rester a distinguer.
    if re.search(r"\bEXO\b.*\b(ALLEG|COTIS)\b|ALLE(G|GE)MENT GENERAL",v):
        return "REDUCTION_MIXTE"
    if re.search(r"\bREDUCTION\b.*\b(AGIRC|ARRCO)\b",v):
        return "RETRAITE"
    if re.search(r"\bREDUCTION (GENERALE|GLE)\b",v):
        return "REDUCTION_A_QUALIFIER"
    if re.search(r"\bREDUCTION SALARIALE HS\b|\bREDUCT HEURES SUPPL\b",v):
        return "URSSAF"

    if re.search(r"\bMUTUELLE\b|\bFRAIS DE SANTE\b",v):
        return "MUTUELLE"
    # Les contributions conventionnelles BTP ont des descriptions
    # distinctes du seul numero de rubrique; maintenir leur regroupement
    # precedent jusqu'a reconciliation avec le tableau des charges.
    # Les caisses de conges payes BTP et leurs contributions associees
    # ne sont pas des cotisations de prevoyance ni des retraites.
    # Le collecteur exact est conserve dans le libelle technique ;
    # la famille de gestion est "Autres" tant qu'elle n'a pas sa ligne.
    if re.search(r"\b(CAISSE DE CONGES PAYES|CCP OPPBTP|BTP ACTION SOCIALE)\b",v):
        return "AUTRES"
    # Assiettes exonerees et non exonerees des apprentis :
    # reconnaitre l'objet, pas les codes locaux EWO0, EW5C, etc.
    if re.search(r"\bVIEILLESSE\b.*\b(PLAF|DEPLAF)\b",v):
        return "URSSAF"
    if re.search(r"\bRET COMPL\b",v):
        return "RETRAITE"
    if "BTP" in v and ("CONSTRUCTYS" in v or "APNAB" in v):
        return "URSSAF" if "TX LEG" in v else "PREVOYANCE"
    if "BTP" in v and "TAXE CCCA" in v:
        return "PREVOYANCE"
    # Garanties de remplacement du revenu et indemnites assurees.
    if re.search(r"^(?:INCAPACITE (?:DE|DU) TRAVAIL|MAINTIEN DE SALAIRE|INDEM(?:NITE)? DEPART RETRAITE(?: (?:TR|TRANCHE) A)?)$",v):
        return "PREVOYANCE"
    if v in {"FONDS DE PEREQUATION","F A P S","FAPS"}:
        return "PREVOYANCE"
    if re.search(r"\bRENTE EDUCATION\b",v):
        return "PREVOYANCE"
    if re.search(r"\bCONTRIBUTION FARE\b|\bCOTISATION FARE\b",v):
        return "AUTRES"
    if re.search(r"\bADESATT\b",v):
        return "AUTRES"
    if re.search(r"\bPARTICIP\b.*\bFORMATION CDD\b|\bCPF CDD\b",v):
        return "URSSAF"
    if re.search(r"\bPREVOYANCE\b|\bPREV\b|\bINCAP\b.*\bINVAL\b|\bIPRIAC\b|\bCFA\b|\bPARITARISME\b|\bFOND ACTION SOCIALE\b",v):
        return "PREVOYANCE"
    if re.search(r"\bAPEC\b|\bRETRAITE COMPL\b|\bCEG\b|\bCET\b|\bAGIRC\b|\bARRCO\b|\bCOMPLEMENTAIRE TRANCHE\b",v):
        return "RETRAITE"
    if re.search(r"\bCHOMAGE\b|\bAGS\b|\bFONDS DE GARANTIE\b",v):
        return "CHOMAGE"
    if re.search(r"\bCSG\b|\bCRDS\b|\bURSSAF\b|\bSEC(U|URITE) SOC\b|\bACC DU TRAV\b|\bMAL PROF\b",v):
        return "URSSAF"
    # Contributions URSSAF par leur objet, pas leur code variable.
    if re.search(r"\b(VERS|VERSEMENT)\b.*\bMOBILITE\b|\bFORFAIT SOCIAL\b",v):
        return "URSSAF"
    if re.search(r"\bREDUCTION\b.*\b(HS|HEURES SUPP|HEURES SUPPLEMENTAIRES)\b",v):
        return "URSSAF"
    if re.search(r"\bMALADIE\b|\bMATERN\b|\bSOLIDARITE AUTONOMIE\b|\bFNAL\b|\bDIALOGUE SOCIAL\b",v):
        return "URSSAF"
    if re.search(r"\bAF TAUX\b|\bAF COMPLEMENT\b|\bALLOCATIONS FAMILIALES\b",v):
        return "URSSAF"
    if re.search(r"\bPARTICIPAT\b.*\bFORMAT\b|\bPARTICIP\b.*\bFORM\b|\bTAXE APPRENTISSAGE\b",v):
        return "URSSAF"
    if v=="FAMILLE" or v.startswith("AUTRES CONTRIB DUES PAR EMPL"):
        return "URSSAF"
    if "COTISATIONS STATUTAIRES" in v and "CC" in v:
        return "AUTRES"
    return None

def extract_employee_id(page_text):
    """Identifiant de paie : chiffres ou lettres/chiffres, sans numéro SS."""
    m=re.search(r"(?im)^\s*Matricule\s*:\s*([A-Za-z0-9][A-Za-z0-9_-]{0,31})(?=\s|$)",page_text)
    return m.group(1) if m else None

def parse(path):
    txt=subprocess.check_output(["pdftotext","-layout",str(path),"-"],timeout=30).decode("utf8")
    pages=[p for p in txt.split("\f") if p.strip()]
    if len(pages)>70:raise ValueError("Trop de pages")
    found={};errors=[]
    for n,p in enumerate(pages,1):
        employee_id=extract_employee_id(p)
        s=re.search(r"Siret\s*:\s*(\d{14})",p)
        d=re.search(r"P[ée]riode\s*:\s*([^\n]+)",p)
        if not (employee_id and s and d):errors.append("Identité incomplète page "+str(n));continue
        key=(s.group(1),d.group(1).strip(),employee_id)
        q=found.setdefault(key,dict(period=d.group(1).strip(),mat=employee_id,nom=None,company=None,pages=[],lines=[],unknown=[],brut=None,pat=None,net=None,pas=None,cout=None,q_layout=False,net_before_tax=None,net_source=None,pas_row_seen=False))
        q["pages"].append(n)
        company_match=re.search(r"(?im)^\s*Soci[eé]t[eé]\s+([^\n]+)",p)
        if company_match and not q["company"]:
            q["company"]=company_match.group(1).strip()
        ll=p.splitlines()
        # PAS fiscal : lire la colonne Montant de la section Impot sur le revenu,
        # jamais la colonne Cumul annuel ni les lignes de cotisations sociales.
        for tax_line in ll:
            if re.search(r"(?i)imp[oô]t sur le revenu pr[eé]lev[eé] [àa] la source",tax_line):
                q["pas_row_seen"]=True
            pas_value=extract_pas(tax_line)
            if pas_value is not None:
                if q["pas"] is not None and q["pas"]!=pas_value:
                    errors.append("PAS incoherent pour matricule "+q["mat"])
                else:
                    q["pas"]=pas_value
        if not q["nom"]:
            for header_line in ll[:38]:
                person_name=re.search(r"(?:^|\s)(?:MME\.?|MLLE\.?|MR\.?|M\.?)[ \t]+([A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]*(?:[ \t][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+)*)(?=\s{2,}|\s*$)",header_line,re.I)
                if person_name:
                    q["nom"]=" ".join(person_name.group(1).split())
                    break
        h=next((x for x in ll if "Montant salarial" in x and "Mt patronal" in x),None)
        if not h:errors.append("Colonnes introuvables page "+str(n));continue
        sx,px=h.index("Montant salarial"),h.index("Mt patronal")
        # Les valeurs finales sont parfois en seconde page, sous le titre.
        for index, line in enumerate(ll):
            if "Versé employeur" in line or "Verse employeur" in line:
                values=[v.group(1) for future in ll[index+1:index+7]
                        for v in [re.fullmatch(r"\s*(-?\d+\.\d{2})\s*",future)] if v]
                if len(values)==1:
                    q["cout"]=cents(values[0])
                elif not values:
                    # Sur certains bulletins BTP, le montant est à droite
                    # de la ligne suivante, en regard d'une autre rubrique.
                    near=ll[index+1:index+3]
                    matches=[m.group() for row in near
                             for m in MONEY.finditer(row)
                             if m.start()>px+20]
                    if len(matches)==1:
                        q["cout"]=cents(matches[0])
            # Montants nets : lecture directe en priorite, repli seulement
            # lorsqu'on dispose du net avant impôt et du PAS explicites.
            printed_net=extract_net_paye(line)
            if printed_net is None:
                printed_net=extract_net_salary_block(ll,index)
            if printed_net is not None:
                if q["net"] is not None and q["net"]!=printed_net:
                    errors.append("Nets contradictoires matricule "+q["mat"])
                else:
                    q["net"]=printed_net
                    q["net_source"]="bulletin"
            before_tax=extract_net_before_tax(line)
            if before_tax is not None:q["net_before_tax"]=before_tax

        # Certains bulletins Quadra sont regroupés par sections Q100-Q802 :
        # les lignes de cotisation n'ont alors aucun code E*/SW* individuel.
        # Les pages de continuation Q801 peuvent commencer sans titre de section.
        q_section="Q801" if q["q_layout"] else None
        q_section_line=re.compile(r"^\s*(Q\d{3})\s+")
        q_labels={
            "Q100":{"Sécu.Soc-Mal.Mater.Inval.Déc.":"URSSAF",
                    "Complémentaire Santé":"PREVOYANCE"},
            "Q200":{"Acc. du trav. - Mal. prof.":"URSSAF"},
            "Q300":{"Sécu.Soc Plafonnée":"URSSAF",
                    "Sécu.Soc Déplafonnée":"URSSAF",
                    "Complémentaire Tranche 1":"RETRAITE"},
            "Q400":{"Famille":"URSSAF"},
            "Q500":{"Chômage":"CHOMAGE","AGS":"CHOMAGE"},
            "Q600":{"Autres contrib. dues par empl.":"URSSAF"},
            "Q800":{"CSG déductible à l'IR":"URSSAF"},
            "Q801":{"CSG/CRDS non déductible à l'IR":"URSSAF"},
            "Q802":{"EXO., ECRET. ET ALLEG. COTIS":"REDUCTION_MIXTE"},
        }
        in_tax_section=False
        for line in ll:
            if "Impôt sur le revenu" in line:
                q_section=None
                in_tax_section=True
            if in_tax_section:
                continue
            section_match=q_section_line.match(line)
            if section_match:
                q_section=section_match.group(1)
                q["q_layout"]=True
                continue
            sal=[v for v in MONEY.finditer(line) if sx-9<=v.start()<px-8]
            pat=[v for v in MONEY.finditer(line) if px-9<=v.start()<=px+14]
            if "SALAIRE BRUT" in line and sal:q["brut"]=cents(sal[-1].group())
            if "TOTAL DES RETENUES" in line and pat:q["pat"]=cents(pat[-1].group())
            # Section de paie terminee : le net et ses explications ne
            # sont pas des lignes de cotisation, meme si le vocabulaire
            # contient "chomage" ou "maladie".
            if normalized_label(line).startswith("DONT EVOLUTION DE LA REMUNERATION"):
                continue
            hit=CODE.match(line)
            rubric=line[:sx].strip().split("  ")[0].strip() if not hit else hit.group(2).strip().split("  ")[0].strip()
            # Les frais de repas, acomptes et remboursements ne sont pas des cotisations.
            if (re.match(r"^\s*(?:S\d{3}|SX\d{2}|BQ[A-Z0-9]+)\s+",line)
                or line[:sx].strip().startswith("Acompte")
                or non_contribution(rubric)):

                continue
            if not hit and (q_section or q["q_layout"]) and (sal or pat):
                # Une même rubrique peut se trouver sous des sections Q
                # différentes selon l'activation du module conventionnel.
                # Dans un bulletin sur deux pages, Q801 peut se poursuivre sans répétition du titre.
                if q_section is None and q["q_layout"] and rubric == "CSG/CRDS non déductible à l\u0027IR":
                    q_section="Q801"
                expected=q_labels.get(q_section,{})
                # Associer le code Q+libellé sans déduire le recouvreur d'un mot isolé.
                category=semantic_group(rubric,q_section) or expected.get(rubric)
                if health_supplement_is_mutuelle(
                        rubric,line[:sx],
                        cents(sal[-1].group()) if sal else 0,
                        cents(pat[-1].group()) if pat else 0):
                    category="MUTUELLE"
                if category and category!="EXCLUDE":
                    q["lines"].append(dict(
                        code=q_section,label=rubric,
                        sal=cents(sal[-1].group()) if sal else 0,
                        pat=cents(pat[-1].group()) if pat else 0,
                        page=n,group=category,
                        base=(cents(MONEY.findall(line[:sx])[0])
                              if MONEY.findall(line[:sx]) else None)))
                    continue
                # Ne pas comptabiliser les totaux ou légendes.
                if rubric and not any(x in rubric.upper() for x in
                   ("TOTAL DES RETENUES","SALAIRE BRUT","NET IMPOSABLE",
                    "COTIS.","NET A PAYER","MONTANT NET SOCIAL",
                    "DONT ÉVOLUTION","IMPÔT SUR LE REVENU")):
                    q["unknown"].append(dict(code=q_section,label=rubric,
                        sal=cents(sal[-1].group()) if sal else 0,
                        pat=cents(pat[-1].group()) if pat else 0,page=n))
                    continue
            if not hit or not (sal or pat) or hit.group(1) in {"BQRE","BPM1","SWZW","SX07","S001"}:continue
            code=hit.group(1)
            item=dict(code=code,label=hit.group(2).strip().split("  ")[0],sal=cents(sal[-1].group()) if sal else 0,pat=cents(pat[-1].group()) if pat else 0,page=n)
            # Le code de rubrique est propre au paramétrage du dossier :
            # il sert au diagnostic, jamais à choisir l'organisme.
            category=semantic_group(item["label"])
            if category=="EXCLUDE":continue
            if category is None:q["unknown"].append(item)
            else:item["group"]=category;q["lines"].append(item)
    for q in found.values():
        # Quadra peut laisser vides les colonnes mensuelles PAS lorsque
        # l'impôt prélevé est nul, tout en montrant "0.00" en cumul annuel.
        # On ne prend jamais ce cumul pour le PAS du mois. Une retenue
        # mensuelle nulle est déduite uniquement si le bulletin imprime
        # à la fois le net payé et le net avant impôt, identiques, et
        # comporte réellement une rubrique d'impôt à la source.
        if infer_zero_pas(q["pas"],q["pas_row_seen"],q["net"],
                          q["net_before_tax"],q["net_source"]):
            q["pas"]=0
        # Cette deduction ne s'active jamais si un NET PAYE est lisible.
        if (q["net"] is None and q["net_before_tax"] is not None
                and q["pas"] is not None):
            q["net"]=q["net_before_tax"]-q["pas"]
            q["net_source"]="calcul net avant PAS - PAS"
        # Un bulletin de solde nul peut ne contenir NI SALAIRE BRUT,
        # NI TOTAL DES RETENUES : ne jamais exiger ces deux lignes.
        # La confirmation vient du net à payer explicitement égal à 0,
        # sans aucune rubrique de cotisation ni brut positif.
        zero_slip=(not q["lines"] and not q["unknown"]
                   and q["net"] == 0 and q["brut"] in (None,0)
                   and q["pat"] in (None,0))
        if zero_slip:
            q["brut"]=0
            q["pat"]=0
            q["cout"]=0
            if q["pas"] is None:q["pas"]=0
        # Test 4 / ancien Quadra : ligne de reduction URSSAF distincte
        # de la ligne de reduction explicitement AGIRC-ARRCO.
        has_explicit_agirc_reduction=any(
            x["group"]=="RETRAITE" and "REDUCTION" in normalized_label(x["label"])
            for x in q["lines"])
        for x in list(q["lines"]):
            if x["group"]=="REDUCTION_A_QUALIFIER":
                if has_explicit_agirc_reduction:
                    x["group"]="URSSAF"
                else:
                    x["group"]="REDUCTION_MIXTE"
        groups=defaultdict(lambda:[0,0])
        for x in q["lines"]:
            # Ne pas inférer le statut conventionnel d'un code E ou EQ.
            # Les prévoyances reconnues sont regroupées par nature.
            g=x["group"]
            groups[g][0]+=x["sal"];groups[g][1]+=x["pat"]
        # Conserver les montants non classés séparément, sans les inventer.
        for x in q["unknown"]:
            groups["A_QUALIFIER"][0]+=x["sal"]
            groups["A_QUALIFIER"][1]+=x["pat"]
        # La ventilation reste indicative lorsque le bulletin simplifié
        # contient seulement la réduction globale.
        q["estimated_split"]=None
        mixed=groups.get("REDUCTION_MIXTE",[0,0])
        mixed_lines=[x for x in q["lines"] if x["group"]=="REDUCTION_MIXTE"]
        if (re.search(r"\b2026\b",q["period"]) and q["q_layout"] and len(mixed_lines)==1
                and mixed_lines[0]["code"]=="Q802" and mixed[0]==0
                and mixed[1]<0):
            reduction=abs(mixed[1])
            # Droit commun 2026 : lire la cotisation patronale de
            # retraite T1, ne jamais déduire ce taux du secteur ou du code.
            # FNAL 0,10 % reste ici une hypothèse de simulation en labo :
            # une estimation n'est PAS un paiement certifié.
            observed_rate=observed_t1_rate(q["lines"])
            standard=estimate_standard_split(
                reduction,observed_rate,fnal_rate="0.10")
            if standard is not None:
                retraite_cents=standard["retraite"]
                urssaf_cents=standard["urssaf"]
                q["split_method"]="simulation 2026 - retraite T1 observée, FNAL 0,10 présumé"
            else:
                # Maintien provisoire de l'ancien résultat sur les
                # répartitions particulières. Ne pas effacer ELM avant de
                # résoudre ses 20,64 EUR d'écart avec la formule officielle.
                retraite_cents=(reduction*54+196)//392
                urssaf_cents=reduction-retraite_cents
                q["split_method"]="historique expérimental - paramètres à vérifier"
            q["observed_t1_rate"]=str(observed_rate) if observed_rate is not None else None
            groups["RETRAITE"][1]-=retraite_cents
            groups["URSSAF"][1]-=urssaf_cents
            del groups["REDUCTION_MIXTE"]
            q["estimated_split"]={"total":reduction,"urssaf":urssaf_cents,
                                  "retraite":retraite_cents}
        q["collector_ambiguity"]=any(
            ("COTISATIONS STATUTAIRES" in normalized_label(x["label"]) or "ADESATT" in normalized_label(x["label"]))
            for x in q["lines"])
        q["groups"]=dict(groups)
        q["delta"]=None if q["pat"] is None else sum(x[1] for x in groups.values())-q["pat"]
    return found,errors
uploaded=st.file_uploader("Importer un PDF de bulletins Quadra du même dossier et du même mois",type="pdf")
if uploaded is None:st.info("Import de bulletins Quadra à texte sélectionnable, sans OCR.");st.stop()
if uploaded.size>15*1024*1024:st.error("PDF trop volumineux");st.stop()
path=None
try:
    with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as t:
        path=Path(t.name);t.write(uploaded.getvalue())
    people,errors=parse(path)
except Exception:
    st.error("Lecture impossible : vérifier le format Quadra.");st.stop()
finally:
    if path and path.exists():path.unlink()
if errors or not people or len(set(x[0] for x in people))!=1 or len(set(x[1] for x in people))!=1:
    st.error("Dossier ou période incohérents, ou lecture incomplète.");st.write(errors);st.stop()
CATEGORIES=["URSSAF","RETRAITE","PREVOYANCE","MUTUELLE","AUTRES"]
LABELS={"URSSAF":"URSSAF","RETRAITE":"Retraite","PREVOYANCE":"Prévoyance","MUTUELLE":"Mutuelle","AUTRES":"Autres"}
def category_totals(person):
    totals=defaultdict(lambda:[0,0])
    for group,amounts in person["groups"].items():
        if group in {"URSSAF","CHOMAGE"}: display="URSSAF"
        elif group=="RETRAITE":display="RETRAITE"
        elif group in {"PREVOYANCE","PREV_COMP","PRO_BTP"}:display="PREVOYANCE"
        elif group=="MUTUELLE":display="MUTUELLE"
        else:display="AUTRES"
        totals[display][0]+=amounts[0]
        totals[display][1]+=amounts[1]
    return totals
# Société : agréger tous les salariés, puis afficher un seul récapitulatif.
company=defaultdict(lambda:[0,0])
for p in people.values():
    for category,amounts in category_totals(p).items():
        company[category][0]+=amounts[0]
        company[category][1]+=amounts[1]

total_brut=sum((p["brut"] or 0) for p in people.values())
total_patronal=sum((p["pat"] or 0) for p in people.values())
total_salarial_classes=sum(v[0] for v in company.values())
total_patronal_classes=sum(v[1] for v in company.values())
has_unknown=any(p["unknown"] for p in people.values())
has_anomalies=any(p["delta"]!=0 for p in people.values())
collector_uncertain=any("REDUCTION_MIXTE" in p["groups"] for p in people.values())
estimated_collector=any(p["estimated_split"] is not None for p in people.values())
ambiguous_collector=any(p["collector_ambiguity"] for p in people.values())
missing_contribution_breakdown=any((p["brut"] or 0)>0 and not (p["lines"] or p["unknown"]) or p["pat"] is None for p in people.values())
contributions_incomplete=has_anomalies or missing_contribution_breakdown
diagnostics=[]
for person in people.values():
    reasons=[]
    if person["unknown"]:reasons.append(str(len(person["unknown"]))+" rubrique(s) inconnue(s)")
    if person["pat"] is None:reasons.append("Total patronal absent")
    elif person["delta"]!=0:reasons.append("Écart patronal : "+euro(person["delta"]))
    if (person["brut"] or 0)>0 and not person["lines"]:reasons.append("Aucune cotisation reconnue")
    if "REDUCTION_MIXTE" in person["groups"]:reasons.append("Réduction non répartie")
    diagnostics.append({"Matricule":person["mat"],
                        "Salarié":person["nom"] or "Non reconnu",
                        "Pages PDF":", ".join(map(str,person["pages"])),
                        "Lignes reconnues":len(person["lines"]),
                        "Lignes inconnues":len(person["unknown"]),
                        "Écart patronal":euro(person["delta"]) if person["delta"] is not None else "Indéterminé",
                        "Statut":" ; ".join(reasons) if reasons else "Rapproché"})
repartition_incomplete=contributions_incomplete

st.subheader("Synthèse de la société")
st.caption("Dossier SIRET : "+next(iter(people))[0]+" · Période : "+next(iter(people))[1])
all_cost_known=all(p["cout"] is not None for p in people.values())
all_net_known=all(p["net"] is not None for p in people.values())
inferred_net=any(p["net_source"]=="calcul net avant PAS - PAS" for p in people.values())
all_pas_known=all(p["pas"] is not None for p in people.values())
total_pas=sum(p["pas"] for p in people.values()) if all_pas_known else None
cout_global=sum(p["cout"] for p in people.values()) if all_cost_known else None
net_global=sum(p["net"] for p in people.values()) if all_net_known else None
st.metric("Coût global des salariés",euro(cout_global) if all_cost_known else "Non déterminé")
a,b,c=st.columns(3)
a.metric("Net total à payer",euro(net_global) if all_net_known else "Non déterminé")
b.metric("Masse salariale brute",euro(total_brut))
c.metric("Cotisations sociales totales",euro(total_salarial_classes+total_patronal_classes) if not contributions_incomplete else "À vérifier")
if all_pas_known:
    st.metric("DGFiP — prélèvement à la source (PAS)",euro(total_pas))
else:
    st.warning("Prélèvement à la source non identifiable pour au moins un salarié.")
st.caption("Cotisations sociales totales : salariales + patronales, hors prélèvement à la source. Les cotisations salariales sont déjà incluses dans les salaires bruts.")
if not all_cost_known or not all_net_known:
    st.warning("Un ou plusieurs montants nets ou coûts employeur ne sont pas identifiables dans les bulletins : les totaux concernés ne sont pas affichés.")
if not all_net_known:
    st.error("Net à payer manquant pour : "+", ".join(
        "matricule "+p["mat"]+" (pages "+", ".join(map(str,p["pages"]))+")"
        for p in people.values() if p["net"] is None)+
        ". Aucune valeur à zéro n'a été inventée. Fournir ce bulletin pour adapter la lecture.")
if inferred_net:
    with st.expander("Contrôle technique du net à payer",expanded=False):
        st.caption("Sur certains bulletins sans ligne NET PAYE exploitable, le net est reconstitué avec le net avant impôt et le prélèvement à la source indiqués sur le même bulletin.")
        st.dataframe([{"Matricule":p["mat"],"Net calculé":euro(p["net"]),
                       "Net avant impôt":euro(p["net_before_tax"]),"PAS":euro(p["pas"])}
                      for p in people.values() if p["net_source"]=="calcul net avant PAS - PAS"],
                     hide_index=True,use_container_width=True)
if any(p["q_layout"] for p in people.values()):
    st.info("Format Q détecté : le regroupement « Complémentaire Santé » et les réductions générales peuvent correspondre à plusieurs organismes. Leur recouvreur exact doit être confirmé.")

if estimated_collector:
    st.caption("Répartition des réductions générales indicative, issue des bulletins. Rapport de gestion, non état officiel de règlement.")
    with st.expander("Notes techniques internes — ventilation de la réduction générale",expanded=False):
        st.warning("Droit commun 2026 : simulation d'après le taux patronal T1 imprimé, avec hypothèse FNAL à 0,10 %. En présence de taux dérogatoires ou de DFS (notamment ELM), l'ancienne estimation reste provisoirement conservée pour ne pas casser les dossiers validés. Les sommes dues par organisme restent à confirmer.")
if contributions_incomplete:
    st.warning("Certaines lignes ne sont pas encore reconnues ou rapprochées. Le diagnostic indique précisément où le moteur bloque.")
elif collector_uncertain:
    st.info("Total social rapproché avec les bulletins, mais la ventilation par organisme reste indisponible : les réductions générales regroupent URSSAF et retraite.")
if all_cost_known and cout_global>0:
    st.subheader("Classement des salariés par coût employeur")
    rank_rows=[{"Salarié":p["nom"] or "Matricule "+p["mat"],"Matricule":p["mat"],
                "Coût employeur":euro(p["cout"]),"Part du coût global":pct(p["cout"],cout_global)}
               for p in sorted(people.values(),key=lambda p:p["cout"],reverse=True)]
    rank_rows.append({"Salarié":"TOTAL","Matricule":"","Coût employeur":euro(cout_global),"Part du coût global":"100,00 %"})
    st.dataframe(rank_rows,hide_index=True,use_container_width=True)
    st.caption("Part du coût global = coût du salarié / coût total employeur × 100. Arrondi à deux décimales.")

if not contributions_incomplete and all_pas_known:
    st.metric("Total à reverser aux organismes et à la DGFiP",
              euro(total_salarial_classes+total_patronal_classes+total_pas))
    st.caption("Ce total inclut les cotisations sociales et le PAS fiscal. Ne pas ajouter une deuxième fois le PAS au coût employeur ni au net déjà prélevé.")

if contributions_incomplete or has_unknown:
    st.subheader("Diagnostic du dossier")
    st.dataframe(diagnostics,hide_index=True,use_container_width=True)
    with st.expander("Rubriques à identifier",expanded=True):
        unknown_rows=[{"Matricule":p["mat"],"Page":x["page"],
                       "Code":x.get("code") or "Sans code","Libellé":x["label"],
                       "Salariales":euro(x["sal"]),"Patronales":euro(x["pat"])}
                      for p in people.values() for x in p["unknown"]]
        if unknown_rows:st.dataframe(unknown_rows,hide_index=True,use_container_width=True)
        else:st.info("Aucune rubrique inconnue : contrôler les totaux et colonnes des pages signalées.")
    st.caption("Les rubriques à qualifier sont incluses dans les montants extraits. Leur nature et leur organisme restent à vérifier.")

# Les familles indépendantes de la réduction générale doivent rester
# visibles même si la ventilation URSSAF / retraite est provisoire.
# Seules les lignes explicitement MUTUELLE / FRAIS DE SANTE entrent ici :
# « Complémentaire Santé » dépend du contrat et de son collecteur.
mutuelle_lignes=[
    (p,x) for p in people.values() for x in p["lines"]
    if x["group"]=="MUTUELLE"]
sante_ambigue=[
    (p,x) for p in people.values() for x in p["lines"]
    if x["group"]=="PREVOYANCE"
    and "COMPLEMENTAIRE SANTE" in normalized_label(x["label"])]
mutuelle_sal=sum(x["sal"] for p,x in mutuelle_lignes)
mutuelle_pat=sum(x["pat"] for p,x in mutuelle_lignes)
mutuelle_matricules=set(p["mat"] for p,x in mutuelle_lignes
                       if x["sal"]!=0 or x["pat"]!=0)
if mutuelle_lignes:
    st.subheader("Mutuelle identifiée dans les bulletins")
    m1,m2,m3=st.columns(3)
    m1.metric("Cotisations salariales",euro(mutuelle_sal))
    m2.metric("Cotisations patronales",euro(mutuelle_pat))
    m3.metric("Total mutuelle identifié",euro(mutuelle_sal+mutuelle_pat))
    st.caption("Mutuelle identifiée sur "+str(len(mutuelle_matricules))+
               " salarié(s), d'après les libellés explicites « Mutuelle » ou « Frais de santé ». "+
               "Ces montants restent visibles même si la répartition URSSAF / retraite est provisoire.")
else:
    st.caption("Aucune cotisation explicitement libellée « Mutuelle » ou « Frais de santé » "+
               "n'a été identifiée. Cela ne prouve pas l'absence de mutuelle dans le dossier.")
if sante_ambigue:
    st.caption("À qualifier séparément : "+str(len(sante_ambigue))+
               " ligne(s) « Complémentaire Santé ». Leur rattachement à la prévoyance "+
               "ou à la mutuelle ne peut pas être déduit du libellé seul.")

# Diagnostic autonome à partir des bulletins : aucune donnée du tableau
# de charges n'est nécessaire pour expliquer une ligne de santé.
health_rows=[{"Matricule":p["mat"],
              "Page":x["page"],
              "Libellé":x["label"],
              "Salariales":euro(x["sal"]),
              "Patronales":euro(x["pat"]),
              "Famille retenue":LABELS.get(x["group"],x["group"])}
             for p in people.values() for x in p["lines"]
             if "SANTE" in normalized_label(x["label"])
             or "MUTUELLE" in normalized_label(x["label"])]
if health_rows:
    with st.expander("Contrôle interne des cotisations santé par salarié",expanded=False):
        st.dataframe(health_rows,hide_index=True,use_container_width=True)
        st.caption("Contrôle fondé sur les libellés, les taux et les montants lus sur les bulletins, sans numéro de rubrique imposé.")

if has_unknown:
    st.warning("Les rubriques à qualifier sont incluses dans Autres. Ventilation indicative, à vérifier avant règlement.")
st.markdown("**Cotisations de toute la société par organisme**")
if contributions_incomplete:
    st.warning("Ventilation non rapprochée : montants extraits ci-dessous, non validés.")
    st.dataframe([{"Organisme":LABELS[c],"Salariales":euro(company[c][0]),"Patronales":euro(company[c][1]),"Total extrait":euro(sum(company[c]))} for c in CATEGORIES],hide_index=True,use_container_width=True)
elif collector_uncertain:
    st.caption("Montants par famille AVANT répartition de la réduction générale commune URSSAF / AGIRC-ARRCO. Ils ne représentent PAS les sommes nettes dues à chaque organisme.")
    reduction_sal=sum(p["groups"].get("REDUCTION_MIXTE",[0,0])[0] for p in people.values())
    reduction_pat=sum(p["groups"].get("REDUCTION_MIXTE",[0,0])[1] for p in people.values())
    provisional=[]
    for cat in CATEGORIES:
        salariale=company[cat][0]-(reduction_sal if cat=="AUTRES" else 0)
        patronale=company[cat][1]-(reduction_pat if cat=="AUTRES" else 0)
        provisional.append({"Organisme":LABELS[cat],"Salariales":euro(salariale),
                            "Patronales":euro(patronale),"Total":euro(salariale+patronale)})
    provisional.append({"Organisme":"Réduction générale à répartir (URSSAF / retraite)",
                        "Salariales":euro(reduction_sal),
                        "Patronales":euro(reduction_pat),
                        "Total":euro(reduction_sal+reduction_pat)})
    provisional.append({"Organisme":"TOTAL DES COTISATIONS",
                        "Salariales":euro(total_salarial_classes),
                        "Patronales":euro(total_patronal_classes),
                        "Total":euro(total_salarial_classes+total_patronal_classes)})
    st.dataframe(provisional,hide_index=True,use_container_width=True)
    st.warning("Les montants nets à verser à l'URSSAF et à la retraite ne sont pas encore déterminés séparément. Ne pas utiliser ce tableau pour payer les organismes.")
else:
    company_social=total_salarial_classes+total_patronal_classes
    company_rows=[{"Organisme":LABELS[c],"Salariales":euro(company[c][0]),
                   "Patronales":euro(company[c][1]),"Total à reverser":euro(sum(company[c])),
                   "Part des cotisations":pct(sum(company[c]),company_social)} for c in CATEGORIES]
    company_rows.append({"Organisme":"TOTAL COTISATIONS SOCIALES","Salariales":euro(total_salarial_classes),
                         "Patronales":euro(total_patronal_classes),
                         "Total à reverser":euro(total_salarial_classes+total_patronal_classes),"Part des cotisations":pct(company_social,company_social)})
    if all_pas_known:
        company_rows.append({"Organisme":"DGFiP — PAS","Salariales":"—",
                             "Patronales":"—","Total à reverser":euro(total_pas),
                             "Part des cotisations":"—"})
        company_rows.append({"Organisme":"TOTAL SOCIAL + PAS","Salariales":"—",
                             "Patronales":"—",
                             "Total à reverser":euro(company_social+total_pas),
                             "Part des cotisations":"—"})
    st.dataframe(company_rows,hide_index=True,use_container_width=True)
    st.caption("Part des cotisations = total de l’organisme / total social de la société, hors PAS. Ne pas confondre avec un taux sur salaire brut.")
    if estimated_collector:
        st.caption("Ventilation indicative : la réduction Q802 est estimée par salarié, avec des écarts possibles de 0,01 € par rapport au détail interne Quadra.")
    st.success("Contrôle patronal : "+euro(total_patronal_classes)+" calculés / "+
               euro(total_patronal)+" sur les bulletins. Écart : 0,00 €.")

st.subheader("Détail par salarié")
for p in people.values():
    with st.expander((p["nom"] or "Nom non reconnu")+" · Matricule "+p["mat"],expanded=len(people)==1):
        g=category_totals(p)
        sal=sum(v[0] for v in g.values())
        pat=sum(v[1] for v in g.values())
        cost_col, share_col=st.columns(2)
        cost_col.metric("Coût employeur du salarié",euro(p["cout"]) if p["cout"] is not None else "Non déterminé")
        share_col.metric("Part du coût total de l’entreprise",pct(p["cout"],cout_global) if p["cout"] is not None else "Non déterminé")
        if p["estimated_split"] is not None:
            x=p["estimated_split"]
            st.info("Réduction générale Q802 estimée : URSSAF "+euro(x["urssaf"])+
                    " · AGIRC-ARRCO "+euro(x["retraite"])+
                    " · Total "+euro(x["total"])+
                    " (ventilation indicative, arrondie au centime).")
            with st.expander("Détail technique de la ventilation",expanded=False):
                st.caption("Méthode : "+p.get("split_method","non déterminée")+
                           " · Taux T1 patronal observé : "+
                           str(p.get("observed_t1_rate") or "non identifié")+" %.")
        a,b,c=st.columns(3)
        a.metric("Net à payer",euro(p["net"]) if p["net"] is not None else "Non déterminé")
        if p["net"] is None:
            st.caption("Net à payer absent de la lecture du PDF (pages : "+", ".join(map(str,p["pages"]))+").")
        b.metric("Salaire brut",euro(p["brut"]) if p["brut"] is not None else "Non déterminé")
        p_incomplete=p["delta"]!=0 or ((p["brut"] or 0)>0 and not (p["lines"] or p["unknown"]))
        p_collector_uncertain="REDUCTION_MIXTE" in p["groups"]
        c.metric("Cotisations sociales totales",euro(sal+pat) if not p_incomplete else "À vérifier")
        if p["pas"] is not None:
            st.metric("DGFiP — prélèvement à la source",euro(p["pas"]))
            if not p_incomplete:
                st.caption("Cotisations + PAS à reverser : "+euro(sal+pat+p["pas"]))
        else:
            st.warning("PAS non identifié sur le bulletin de ce salarié.")
        if not p_incomplete and p["brut"] and p["brut"]>0:
            st.metric("Taux de cotisations patronales / brut",pct(pat,p["brut"]))
            st.caption("Taux patronal = cotisations patronales nettes après réductions / salaire brut × 100.")
        if p_incomplete:
            st.warning("Montants extraits non rapprochés avec le bulletin.")
            st.dataframe([{"Organisme":LABELS[c],"Salariales":euro(g[c][0]),"Patronales":euro(g[c][1]),"Total extrait":euro(sum(g[c]))} for c in CATEGORIES],hide_index=True,use_container_width=True)
        elif p_collector_uncertain:
            # Le code de réduction commune est suivi indépendamment pour ne pas
            # l'imputer arbitrairement à l'URSSAF ou à la retraite.
            mixed=p["groups"].get("REDUCTION_MIXTE",[0,0])
            preview=[]
            for cat in CATEGORIES:
                salariale=g[cat][0]-(mixed[0] if cat=="AUTRES" else 0)
                patronale=g[cat][1]-(mixed[1] if cat=="AUTRES" else 0)
                preview.append({"Famille":LABELS[cat],"Salariales":euro(salariale),
                                "Patronales":euro(patronale),"Total":euro(salariale+patronale)})
            preview.append({"Famille":"Réduction générale à répartir",
                            "Salariales":euro(mixed[0]),"Patronales":euro(mixed[1]),
                            "Total":euro(sum(mixed))})
            preview.append({"Famille":"TOTAL","Salariales":euro(sal),
                            "Patronales":euro(pat),"Total":euro(sal+pat)})
            st.dataframe(preview,hide_index=True,use_container_width=True)
            st.caption("Ventilation indicative avant partage de la réduction générale : pas un état de règlement par organisme.")
        else:
            detail=[{"Organisme":LABELS[c],"Salariales":euro(g[c][0]),
                     "Patronales":euro(g[c][1]),"Total à reverser":euro(sum(g[c])),
                     "Part des cotisations":pct(sum(g[c]),sal+pat)} for c in CATEGORIES]
            detail.append({"Organisme":"TOTAL COTISATIONS SOCIALES","Salariales":euro(sal),
                           "Patronales":euro(pat),"Total à reverser":euro(sal+pat),"Part des cotisations":pct(sal+pat,sal+pat)})
            if p["pas"] is not None:
                detail.append({"Organisme":"DGFiP — PAS","Salariales":"—",
                               "Patronales":"—","Total à reverser":euro(p["pas"]),
                               "Part des cotisations":"—"})
                detail.append({"Organisme":"TOTAL SOCIAL + PAS","Salariales":"—",
                               "Patronales":"—",
                               "Total à reverser":euro(sal+pat+p["pas"]),
                               "Part des cotisations":"—"})
            st.dataframe(detail,hide_index=True,use_container_width=True)
            st.caption("Part des cotisations = cotisations de cet organisme / cotisations totales du salarié. « — » si le total est nul.")
        if p["delta"]==0 and not p_incomplete:
            st.caption("Contrôle patronal : "+euro(pat)+" calculés / "+
                       euro(p["pat"] or 0)+" sur le bulletin · Écart : 0,00 €")
        else:
            st.warning("Écart patronal à contrôler : "+
                       (euro(p["delta"]) if p["delta"] is not None else "indéterminé"))
        if p["unknown"]:
            st.warning(str(len(p["unknown"]))+" rubrique(s) à qualifier ; montants inclus provisoirement.")
            with st.expander("Anomalies techniques"):
                st.dataframe([{**x,"sal":euro(x["sal"]),"pat":euro(x["pat"])} for x in p["unknown"]],hide_index=True,use_container_width=True)

st.subheader("Rapport client PDF")
pdf_allowed=(not contributions_incomplete
             and all_cost_known and all_net_known and all_pas_known)
if pdf_allowed:
    from report_pdf import make_report
    pdf_employees=[]
    for person in people.values():
        g=category_totals(person)
        pdf_employees.append({
            "name":person["nom"] or "Matricule "+person["mat"],
            "mat":person["mat"],"gross":person["brut"] or 0,
            "net":person["net"],"pas":person["pas"],"cost":person["cout"],
            "sal":sum(x[0] for x in g.values()),
            "pat":sum(x[1] for x in g.values()),
            "groups":{key:list(g[key]) for key in CATEGORIES},
        })
    try:
        pdf_bytes=make_report(next(iter(people))[0],next(iter(people))[1],
                              pdf_employees,estimated=estimated_collector or collector_uncertain,
                              provisional=has_unknown or collector_uncertain or ambiguous_collector,
                              company_name=next(iter(people.values())).get("company"))
        st.download_button("Télécharger le rapport PDF",data=pdf_bytes,
                           file_name="paiedyali_rapport_entreprise.pdf",
                           mime="application/pdf",use_container_width=True)
        st.caption("Rapport de gestion fondé sur les bulletins importés. Vérifier les résultats avant transmission au client.")
    except Exception:
        st.error("Impossible de préparer le rapport PDF. Vérifier le modèle de rapport.")
else:
    st.warning("Rapport client PDF indisponible : montants non rapprochés ou données essentielles manquantes.")

if repartition_incomplete:
    st.caption("L'export des montants par organisme est désactivé tant que les cotisations ne sont pas rapprochées.")
else:
    # Export local de la synthèse : aucune base historique n'est créée.
    import csv
    import io
    csv_buffer=io.StringIO()
    csv_writer=csv.writer(csv_buffer,delimiter=";")
    csv_writer.writerow(["Période","Salarié","Matricule","Brut","Net à payer","DGFiP PAS",
                         "Coût employeur","Cotisations salariales","Cotisations patronales",
                         "Cotisations totales","URSSAF","Retraite","Prévoyance","Mutuelle","Autres",
                         "Statut du contrôle"])
    for person in people.values():
        categorized=category_totals(person)
        sal_total=sum(x[0] for x in categorized.values())
        pat_total=sum(x[1] for x in categorized.values())
        csv_writer.writerow([
            next(iter(people))[1],person["nom"] or "Nom non reconnu",person["mat"],
            euro(person["brut"]) if person["brut"] is not None else "Non déterminé",
            euro(person["net"]) if person["net"] is not None else "Non déterminé",
            euro(person["pas"]) if person["pas"] is not None else "Non déterminé",
            euro(person["cout"]) if person["cout"] is not None else "Non déterminé",
            euro(sal_total),euro(pat_total),euro(sal_total+pat_total),
            *[euro(sum(categorized[category])) for category in CATEGORIES],
            "Ventilation provisoire" if person["unknown"] or "REDUCTION_MIXTE" in person["groups"] or person["estimated_split"] is not None or person["collector_ambiguity"] else "Patronal rapproché"
        ])
    st.download_button("Télécharger le récapitulatif CSV des salariés",
                       data="\ufeff"+csv_buffer.getvalue(),
                       file_name="paiedyali_entreprise_recap.csv",
                       mime="text/csv",use_container_width=True)
    st.caption("Fichier de test : données issues des PDF importés, à vérifier avant utilisation comptable.")

st.caption("URSSAF affichée inclut chômage et AGS. Le tableau des charges reste nécessaire pour valider les rattachements expérimentaux, notamment PRO BTP.")


# Alerte de couverture retraite, sans inventer de cotisations.
retirement_missing=[
    p for p in people.values()
    if (p["brut"] or 0)>0
    and any(x["group"]=="URSSAF" and (x["sal"] or x["pat"]) for x in p["lines"])
    and not any(x["group"]=="RETRAITE" and
                "REDUCTION" not in normalized_label(x["label"])
                for x in p["lines"])
]
if retirement_missing:
    st.warning("Retraite complémentaire : aucune cotisation identifiée pour les matricules "+
               ", ".join(p["mat"] for p in retirement_missing)+
               ". Vérifier l'affiliation et le paramétrage AGIRC-ARRCO, sous réserve des situations particulières. "
               "Cette alerte ne bloque pas les exports.")

# Rappel juridique informatif, indépendant des calculs et du rapprochement.
absence_mutuelle=not any(mutuelle_lignes)
absence_prevoyance=not any(
    x["group"] in {"PREVOYANCE","PREV_COMP","PRO_BTP"}
    for p in people.values() for x in p["lines"])
if absence_mutuelle or absence_prevoyance:
    st.subheader("Rappel légal — protection sociale complémentaire")
    if absence_mutuelle:
        st.warning("Mutuelle : aucune rubrique explicite identifiée. Dans le secteur privé, "
                   "l'employeur doit en principe proposer une couverture santé collective et "
                   "financer au moins 50 % de la couverture obligatoire, sous réserve des exceptions "
                   "et dispenses applicables. Vérifier les contrats et justificatifs. "
                   "Article L. 911-7 du Code de la sécurité sociale.")
        st.markdown("[Référence officielle — complémentaire santé](https://entreprendre.service-public.gouv.fr/vosdroits/F33754)")
    if absence_prevoyance:
        st.warning("Prévoyance : aucune rubrique identifiée. Vérifier les obligations de la convention "
                   "collective et des accords applicables. Pour les cadres et assimilés concernés, "
                   "l'ANI du 17 novembre 2017 prévoit une contribution patronale exclusive de 1,50 % "
                   "sur la tranche de rémunération sous le plafond de la Sécurité sociale, "
                   "affectée prioritairement au risque décès.")
        st.markdown("[Référence officielle — prévoyance des cadres](https://www.legifrance.gouv.fr/conv_coll/article/KALIARTI000036732016)")
    st.caption("L'absence de rubrique identifiée ne prouve ni l'absence de couverture ni un manquement. "
               "Ce rappel ne constitue pas un diagnostic juridique et ne modifie aucun montant.")
