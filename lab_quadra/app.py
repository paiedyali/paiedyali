import os,re,hmac,tempfile,subprocess
from pathlib import Path
from decimal import Decimal
from collections import defaultdict
import streamlit as st
st.set_page_config(page_title="PaieDyali Entreprise - laboratoire Quadra",layout="wide")
st.title("PaieDyali Entreprise - Laboratoire Quadra")
st.caption("Réservé aux bulletins fictifs de test. Ventilations expérimentales, non certifiées.")
secret=os.getenv("PAIEDYALI_TEST_PASSWORD","")
if not secret:st.error("Mot de passe non configuré");st.stop()
pw=st.text_input("Mot de passe du laboratoire",type="password")
if not pw or not hmac.compare_digest(pw,secret):
    if pw:st.error("Accès refusé")
    st.stop()
MAP={}
for group,codes in {
"URSSAF":"EQ01 EW7A EQ16 EQ03 EQZ0 EQZ4 EQ05 EQ06 EQ07 EQ5N EQF7 EW27 SW01 SW02 SWB1 SWB3 SWB4 SWB6 EWID EWIF SWA2 SWA4 EW5R EW6X EW7S EW70 EW77 EW79 EW7V EWZB EWA2",
"CHOMAGE":"EWK1 EQ19","RETRAITE":"EW5A EW6A EWF7 EQ1R",
"PREVOYANCE":"E002 E006 E007 E_EH E_VQ EQH7 EQH8 EQH9 EQT7",
"MUTUELLE":"E003 E004",
"PRO_BTP":"EQH2 EQ8O EQDG EWHO EQHR EQHV"}.items():
    for code in codes.split():MAP[code]=group
MONEY=re.compile(r"-?\d+\.\d{2,4}")
CODE=re.compile(r"^\s*((?:E[0-9]{3}|(?:EQ|EW|SW|E_)[A-Z0-9_]{2,3}))\s+(.+)$")
def cents(s):return int((Decimal(s)*100).quantize(Decimal("1")))
def euro(c):return f"{c/100:,.2f} €".replace(","," ").replace(".",",")
def pct(part,base):
    if base is None or base<=0:return "—"
    return f"{part*100/base:.2f}".replace(".",",")+" %"
def parse(path):
    txt=subprocess.check_output(["pdftotext","-layout",str(path),"-"],timeout=30).decode("utf8")
    pages=[p for p in txt.split("\f") if p.strip()]
    if len(pages)>70:raise ValueError("Trop de pages")
    found={};errors=[]
    for n,p in enumerate(pages,1):
        m=re.search(r"Matricule\s*:\s*(\d+)",p)
        s=re.search(r"Siret\s*:\s*(\d{14})",p)
        d=re.search(r"P[ée]riode\s*:\s*([^\n]+)",p)
        if not (m and s and d):errors.append("Identité incomplète page "+str(n));continue
        key=(s.group(1),d.group(1).strip(),m.group(1))
        q=found.setdefault(key,dict(mat=m.group(1),nom=None,pages=[],lines=[],unknown=[],brut=None,pat=None,net=None,cout=None,q_layout=False))
        q["pages"].append(n)
        ll=p.splitlines()
        if not q["nom"]:
            for header_line in ll[:38]:
                person_name=re.search(r"(?:^|\s)(?:MME|Mme|Mlle|M\.)\s+([A-ZÀ-ÖØ-Þ][A-ZÀ-ÖØ-Þ\x27 -]{2,})",header_line)
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
            if "NET PAYE" in line:
                net_paye=re.search(r"NET PAYE\s+(-?\d+\.\d{2})\s+Euros",line)
                if net_paye:q["net"]=cents(net_paye.group(1))
            if "Net à payer au salarié" in line:
                values=[v.group(1) for future in ll[index+1:index+8]
                        for v in [re.fullmatch(r"\s*(-?\d+\.\d{2})\s*",future)] if v]
                if len(values)==1:q["net"]=cents(values[0])

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
            hit=CODE.match(line)
            # Les frais de repas, acomptes et remboursements ne sont pas des cotisations.
            if (re.match(r"^\s*(?:S\d{3}|SX\d{2}|BQ[A-Z0-9]+)\s+",line)
                or line[:sx].strip().startswith("Acompte")):
                continue
            if not hit and (q_section or q["q_layout"]) and (sal or pat):
                # Accepter une rubrique Q seulement si son libellé est explicitement connu.
                rubric=line[:sx].strip().split('  ')[0].strip()
                # Dans un bulletin sur deux pages, Q801 peut se poursuivre sans répétition du titre.
                if q_section is None and q["q_layout"] and rubric == "CSG/CRDS non déductible à l\u0027IR":
                    q_section="Q801"
                expected=q_labels.get(q_section,{})
                # Associer le code Q+libellé sans déduire le recouvreur d'un mot isolé.
                if rubric in expected:
                    q["lines"].append(dict(
                        code=q_section,label=rubric,
                        sal=cents(sal[-1].group()) if sal else 0,
                        pat=cents(pat[-1].group()) if pat else 0,
                        page=n,group=expected[rubric]))
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
            if code not in MAP:q["unknown"].append(item)
            else:item["group"]=MAP[code];q["lines"].append(item)
    for q in found.values():
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
        ouvrier=any(x["code"] in {"EQH7","EQH8","EQH9","EQT7"} for x in q["lines"])
        groups=defaultdict(lambda:[0,0])
        for x in q["lines"]:
            g=x["group"]
            if x["code"] in {"EQDG","EWHO","EQHR","EQHV"} and ouvrier:g="PREV_COMP"
            x["group"]=g
            groups[g][0]+=x["sal"];groups[g][1]+=x["pat"]
        # Hypothèse expérimentale validée sur ELM 09/2026 :
        # clé AGIRC-ARRCO 5,40 / 39,20 ; solde à l'URSSAF.
        # Il s'agit d'une ventilation estimée, PAS d'un taux universel ni
        # d'une validation des sommes exigibles par les organismes.
        q["estimated_split"]=None
        mixed=groups.get("REDUCTION_MIXTE",[0,0])
        mixed_lines=[x for x in q["lines"] if x["group"]=="REDUCTION_MIXTE"]
        if (q["q_layout"] and len(mixed_lines)==1
                and mixed_lines[0]["code"]=="Q802" and mixed[0]==0
                and mixed[1]<0):
            reduction=abs(mixed[1])
            retraite_cents=(reduction*54+196)//392
            urssaf_cents=reduction-retraite_cents
            groups["RETRAITE"][1]-=retraite_cents
            groups["URSSAF"][1]-=urssaf_cents
            del groups["REDUCTION_MIXTE"]
            q["estimated_split"]={"total":reduction,"urssaf":urssaf_cents,
                                  "retraite":retraite_cents}
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
has_anomalies=any(p["unknown"] or p["delta"]!=0 for p in people.values())
collector_uncertain=any("REDUCTION_MIXTE" in p["groups"] for p in people.values())
estimated_collector=any(p["estimated_split"] is not None for p in people.values())
missing_contribution_breakdown=any((p["brut"] or 0)>0 and not p["lines"] or p["pat"] is None for p in people.values())
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
repartition_incomplete=contributions_incomplete or collector_uncertain or estimated_collector

st.subheader("Synthèse de la société")
st.caption("Dossier SIRET : "+next(iter(people))[0]+" · Période : "+next(iter(people))[1])
all_cost_known=all(p["cout"] is not None for p in people.values())
all_net_known=all(p["net"] is not None for p in people.values())
cout_global=sum(p["cout"] for p in people.values()) if all_cost_known else None
net_global=sum(p["net"] for p in people.values()) if all_net_known else None
st.metric("Coût global des salariés",euro(cout_global) if all_cost_known else "Non déterminé")
a,b,c=st.columns(3)
a.metric("Net total à payer",euro(net_global) if all_net_known else "Non déterminé")
b.metric("Masse salariale brute",euro(total_brut))
c.metric("Cotisations sociales totales",euro(total_salarial_classes+total_patronal_classes) if not contributions_incomplete else "À vérifier")
st.caption("Cotisations sociales totales : salariales + patronales, hors prélèvement à la source. Les cotisations salariales sont déjà incluses dans les salaires bruts.")
if not all_cost_known or not all_net_known:
    st.warning("Un ou plusieurs montants nets ou coûts employeur ne sont pas identifiables dans les bulletins : les totaux concernés ne sont pas affichés.")
if any(p["q_layout"] for p in people.values()):
    st.info("Format Q détecté : le regroupement « Complémentaire Santé » et les réductions générales peuvent correspondre à plusieurs organismes. Leur recouvreur exact doit être confirmé.")

if estimated_collector:
    st.warning("Répartition URSSAF / AGIRC-ARRCO estimée par la clé expérimentale 5,40 / 39,20 sur les lignes Q802. Validée globalement sur ELM 09/2026, non certifiée sur d'autres dossiers. Ne pas utiliser ces montants pour payer les organismes.")
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

if contributions_incomplete:
    st.subheader("Diagnostic du dossier")
    st.dataframe(diagnostics,hide_index=True,use_container_width=True)
    with st.expander("Rubriques à identifier",expanded=True):
        unknown_rows=[{"Matricule":p["mat"],"Page":x["page"],
                       "Code":x.get("code") or "Sans code","Libellé":x["label"],
                       "Salariales":euro(x["sal"]),"Patronales":euro(x["pat"])}
                      for p in people.values() for x in p["unknown"]]
        if unknown_rows:st.dataframe(unknown_rows,hide_index=True,use_container_width=True)
        else:st.info("Aucune rubrique inconnue : contrôler les totaux et colonnes des pages signalées.")
    st.caption("Les salariés rapprochés restent consultables individuellement. Les totaux société ne sont pas présentés comme validés.")

st.markdown("**Cotisations de toute la société par organisme**")
if contributions_incomplete:
    st.warning("Ventilation société non validée. Voir le diagnostic ci-dessus et les salariés rapprochés ci-dessous.")
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
    company_rows.append({"Organisme":"TOTAL","Salariales":euro(total_salarial_classes),
                         "Patronales":euro(total_patronal_classes),
                         "Total à reverser":euro(total_salarial_classes+total_patronal_classes),"Part des cotisations":pct(company_social,company_social)})
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
                    " (clé expérimentale 5,40 / 39,20, arrondi au centime).")
        a,b,c=st.columns(3)
        a.metric("Net à payer",euro(p["net"]) if p["net"] is not None else "Non déterminé")
        b.metric("Salaire brut",euro(p["brut"]) if p["brut"] is not None else "Non déterminé")
        p_incomplete=bool(p["unknown"]) or p["delta"]!=0 or ((p["brut"] or 0)>0 and not p["lines"])
        p_collector_uncertain="REDUCTION_MIXTE" in p["groups"]
        c.metric("Cotisations sociales totales",euro(sal+pat) if not p_incomplete else "À vérifier")
        if not p_incomplete and p["brut"] and p["brut"]>0:
            st.metric("Taux de cotisations patronales / brut",pct(pat,p["brut"]))
            st.caption("Taux patronal = cotisations patronales nettes après réductions / salaire brut × 100.")
        if p_incomplete:
            st.error("Ventilation indisponible : des cotisations sont non reconnues ou ne se rapprochent pas du bulletin.")
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
            detail.append({"Organisme":"TOTAL","Salariales":euro(sal),
                           "Patronales":euro(pat),"Total à reverser":euro(sal+pat),"Part des cotisations":pct(sal+pat,sal+pat)})
            st.dataframe(detail,hide_index=True,use_container_width=True)
            st.caption("Part des cotisations = cotisations de cet organisme / cotisations totales du salarié. « — » si le total est nul.")
        if p["delta"]==0 and not p_incomplete:
            st.caption("Contrôle patronal : "+euro(pat)+" calculés / "+
                       euro(p["pat"] or 0)+" sur le bulletin · Écart : 0,00 €")
        else:
            st.warning("Écart patronal à contrôler : "+
                       (euro(p["delta"]) if p["delta"] is not None else "indéterminé"))
        if p["unknown"]:
            st.error(str(len(p["unknown"]))+" rubrique(s) non classée(s).")
            with st.expander("Anomalies techniques"):
                st.dataframe(p["unknown"],hide_index=True,use_container_width=True)

if repartition_incomplete:
    st.caption("L'export des montants par organisme est désactivé tant que les cotisations ne sont pas rapprochées.")
else:
    # Export local de la synthèse : aucune base historique n'est créée.
    import csv
    import io
    csv_buffer=io.StringIO()
    csv_writer=csv.writer(csv_buffer,delimiter=";")
    csv_writer.writerow(["Période","Salarié","Matricule","Brut","Net à payer",
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
            euro(person["cout"]) if person["cout"] is not None else "Non déterminé",
            euro(sal_total),euro(pat_total),euro(sal_total+pat_total),
            *[euro(sum(categorized[category])) for category in CATEGORIES],
            "A vérifier" if person["unknown"] or person["delta"]!=0 else "Patronal rapproché"
        ])
    st.download_button("Télécharger le récapitulatif CSV des salariés",
                       data="\ufeff"+csv_buffer.getvalue(),
                       file_name="paiedyali_entreprise_recap.csv",
                       mime="text/csv",use_container_width=True)
    st.caption("Fichier de test : données issues des PDF importés, à vérifier avant utilisation comptable.")

st.caption("URSSAF affichée inclut chômage et AGS. Le tableau des charges reste nécessaire pour valider les rattachements expérimentaux, notamment PRO BTP.")
