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
        q=found.setdefault(key,dict(mat=m.group(1),nom=None,pages=[],lines=[],unknown=[],brut=None,pat=None,net=None,cout=None))
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
            if "Net à payer au salarié" in line:
                values=[v.group(1) for future in ll[index+1:index+8]
                        for v in [re.fullmatch(r"\s*(-?\d+\.\d{2})\s*",future)] if v]
                if len(values)==1:q["net"]=cents(values[0])

        for line in ll:
            sal=[v for v in MONEY.finditer(line) if sx-9<=v.start()<px-8]
            pat=[v for v in MONEY.finditer(line) if px-9<=v.start()<=px+10]
            if "SALAIRE BRUT" in line and sal:q["brut"]=cents(sal[-1].group())
            if "TOTAL DES RETENUES" in line and pat:q["pat"]=cents(pat[-1].group())
            hit=CODE.match(line)
            if not hit or not (sal or pat) or hit.group(1) in {"BQRE","BPM1","SWZW","SX07","S001"}:continue
            code=hit.group(1)
            item=dict(code=code,label=hit.group(2).strip().split("  ")[0],sal=cents(sal[-1].group()) if sal else 0,pat=cents(pat[-1].group()) if pat else 0,page=n)
            if code not in MAP:q["unknown"].append(item)
            else:item["group"]=MAP[code];q["lines"].append(item)
    for q in found.values():
        if not q["lines"] and q["pat"] is None:q["pat"]=0;q["brut"]=q["brut"] or 0
        # Un bulletin de paie nul n'a pas de montant "Versé employeur" imprimé.
        # Quadra laisse ce champ vide ; le coût est bien zéro dans ce cas.
        if q["cout"] is None and q["brut"] == 0 and q["pat"] == 0:
            q["cout"]=0
        ouvrier=any(x["code"] in {"EQH7","EQH8","EQH9","EQT7"} for x in q["lines"])
        groups=defaultdict(lambda:[0,0])
        for x in q["lines"]:
            g=x["group"]
            if x["code"] in {"EQDG","EWHO","EQHR","EQHV"} and ouvrier:g="PREV_COMP"
            x["group"]=g
            groups[g][0]+=x["sal"];groups[g][1]+=x["pat"]
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
c.metric("Cotisations sociales totales",euro(total_salarial_classes+total_patronal_classes))
st.caption("Cotisations sociales totales : salariales + patronales, hors prélèvement à la source. Les cotisations salariales sont déjà incluses dans les salaires bruts.")
if not all_cost_known or not all_net_known:
    st.warning("Un ou plusieurs montants nets ou coûts employeur ne sont pas identifiables dans les bulletins : les totaux concernés ne sont pas affichés.")

st.markdown("**Cotisations de toute la société par organisme**")
company_rows=[{"Organisme":LABELS[c],"Salariales":euro(company[c][0]),
               "Patronales":euro(company[c][1]),"Total à reverser":euro(sum(company[c]))} for c in CATEGORIES]
company_rows.append({"Organisme":"TOTAL","Salariales":euro(total_salarial_classes),
                     "Patronales":euro(total_patronal_classes),
                     "Total à reverser":euro(total_salarial_classes+total_patronal_classes)})
st.dataframe(company_rows,hide_index=True,use_container_width=True)

if has_anomalies:
    st.warning("Résultat provisoire : cotisations non reconnues ou écart patronal sur au moins un bulletin.")
else:
    st.success("Contrôle patronal : "+euro(total_patronal_classes)+" calculés / "+
               euro(total_patronal)+" sur les bulletins. Écart : 0,00 €.")

# Inspiration PayDashboard : comparer deux jeux de bulletins, sans stockage serveur.
st.subheader("Comparer deux périodes")
st.caption("Importez en option les bulletins d'une autre période du même SIRET. Le calcul se fait uniquement pendant cette session.")
reference_file=st.file_uploader("PDF de référence (autre mois, même entreprise)",type="pdf",key="comparaison")
if reference_file is not None:
    reference_path=None
    if reference_file.size>15*1024*1024:
        st.error("Le PDF de référence dépasse 15 Mo.")
    else:
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as tmp:
                reference_path=Path(tmp.name)
                tmp.write(reference_file.getvalue())
            reference,reference_errors=parse(reference_path)
        except Exception:
            st.error("Le PDF de référence n'a pas pu être lu dans le format Quadra attendu.")
            reference=None
            reference_errors=[]
        finally:
            if reference_path is not None and reference_path.exists():
                reference_path.unlink()
        if reference is not None:
            current_siret=next(iter(people))[0]
            previous_sirets={k[0] for k in reference}
            previous_periods={k[1] for k in reference}
            current_period=next(iter(people))[1]
            if reference_errors or not reference:
                st.error("Référence incomplète : aucune comparaison affichée.")
                st.write(reference_errors)
            elif previous_sirets!={current_siret} or len(previous_periods)!=1:
                st.error("La comparaison exige un seul établissement (même SIRET) et une seule période par PDF.")
            elif current_period in previous_periods:
                st.error("Les deux PDF concernent la même période. Importez une autre période.")
            else:
                reference_period=next(iter(previous_periods))
                old_people={p["mat"]:p for p in reference.values()}
                new_people={p["mat"]:p for p in people.values()}
                old_group=defaultdict(lambda:[0,0])
                for old_person in reference.values():
                    for group,values in category_totals(old_person).items():
                        old_group[group][0]+=values[0]
                        old_group[group][1]+=values[1]
                # Toutes les mesures affichées sont calculées uniquement depuis les PDF importés.
                old_brut=sum((p["brut"] or 0) for p in reference.values())
                old_social=sum(sum(v) for v in old_group.values())
                old_net=(sum(p["net"] for p in reference.values())
                         if all(p["net"] is not None for p in reference.values()) else None)
                old_cost=(sum(p["cout"] for p in reference.values())
                          if all(p["cout"] is not None for p in reference.values()) else None)
                current_social=total_salarial_classes+total_patronal_classes
                st.markdown("**Période analysée :** "+current_period+" · **Référence :** "+reference_period)
                comparison=[
                    {"Indicateur":"Coût global","Référence":euro(old_cost) if old_cost is not None else "Non déterminé",
                     "Période analysée":euro(cout_global) if cout_global is not None else "Non déterminé",
                     "Évolution":euro(cout_global-old_cost) if cout_global is not None and old_cost is not None else "Non calculable"},
                    {"Indicateur":"Net total","Référence":euro(old_net) if old_net is not None else "Non déterminé",
                     "Période analysée":euro(net_global) if net_global is not None else "Non déterminé",
                     "Évolution":euro(net_global-old_net) if net_global is not None and old_net is not None else "Non calculable"},
                    {"Indicateur":"Masse salariale brute","Référence":euro(old_brut),
                     "Période analysée":euro(total_brut),"Évolution":euro(total_brut-old_brut)},
                    {"Indicateur":"Cotisations sociales","Référence":euro(old_social),
                     "Période analysée":euro(current_social),"Évolution":euro(current_social-old_social)}
                ]
                st.dataframe(comparison,hide_index=True,use_container_width=True)
                st.markdown("**Évolution des cotisations par organisme**")
                variations=[]
                for group in CATEGORIES:
                    before=sum(old_group[group])
                    now=sum(company[group])
                    variations.append({"Organisme":LABELS[group],"Référence":euro(before),
                                       "Période analysée":euro(now),"Écart":euro(now-before)})
                st.dataframe(variations,hide_index=True,use_container_width=True)
                st.markdown("**Évolution par salarié (identification par matricule)**")
                changes=[]
                for mat in sorted(set(old_people)|set(new_people),key=lambda x:(not x.isdigit(),int(x) if x.isdigit() else x)):
                    old=old_people.get(mat)
                    new=new_people.get(mat)
                    label=(new or old)["nom"] or "Nom non reconnu"
                    if old is None: status="Présent uniquement sur la période analysée"
                    elif new is None:status="Présent uniquement sur la référence"
                    else:status="Présent sur les deux périodes"
                    old_value=old["cout"] if old else None
                    new_value=new["cout"] if new else None
                    changes.append({"Salarié":label,"Matricule":mat,"Situation":status,
                                    "Coût référence":euro(old_value) if old_value is not None else "—",
                                    "Coût période":euro(new_value) if new_value is not None else "—",
                                    "Variation":euro(new_value-old_value) if new_value is not None and old_value is not None else "—"})
                st.dataframe(changes,hide_index=True,use_container_width=True)
                uncertain_old=any(p["delta"]!=0 or p["unknown"] for p in reference.values())
                if uncertain_old or has_anomalies:
                    st.warning("Attention : au moins une période présente des écarts ou rubriques inconnues. La comparaison est indicative.")
                st.caption("Une variation ne prouve pas sa cause. Les arrivées, départs, absences, primes, régularisations et frais doivent être analysés avant toute explication. Aucun historique n'est conservé dans cette version du laboratoire.")

st.subheader("Détail par salarié")
for p in people.values():
    with st.expander((p["nom"] or "Nom non reconnu")+" · Matricule "+p["mat"],expanded=len(people)==1):
        g=category_totals(p)
        sal=sum(v[0] for v in g.values())
        pat=sum(v[1] for v in g.values())
        st.metric("Coût total du salarié",euro(p["cout"]) if p["cout"] is not None else "Non déterminé")
        a,b,c=st.columns(3)
        a.metric("Net à payer",euro(p["net"]) if p["net"] is not None else "Non déterminé")
        b.metric("Salaire brut",euro(p["brut"]) if p["brut"] is not None else "Non déterminé")
        c.metric("Cotisations sociales totales",euro(sal+pat))
        detail=[{"Organisme":LABELS[c],"Salariales":euro(g[c][0]),
                 "Patronales":euro(g[c][1]),"Total à reverser":euro(sum(g[c]))} for c in CATEGORIES]
        detail.append({"Organisme":"TOTAL","Salariales":euro(sal),
                       "Patronales":euro(pat),"Total à reverser":euro(sal+pat)})
        st.dataframe(detail,hide_index=True,use_container_width=True)
        if p["delta"]==0:
            st.caption("Contrôle patronal : "+euro(pat)+" calculés / "+
                       euro(p["pat"] or 0)+" sur le bulletin · Écart : 0,00 €")
        else:
            st.warning("Écart patronal à contrôler : "+
                       (euro(p["delta"]) if p["delta"] is not None else "indéterminé"))
        if p["unknown"]:
            st.error(str(len(p["unknown"]))+" rubrique(s) non classée(s).")
            with st.expander("Anomalies techniques"):
                st.dataframe(p["unknown"],hide_index=True,use_container_width=True)

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
