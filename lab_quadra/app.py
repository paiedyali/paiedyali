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
        q=found.setdefault(key,dict(mat=m.group(1),pages=[],lines=[],unknown=[],brut=None,pat=None))
        q["pages"].append(n)
        ll=p.splitlines()
        h=next((x for x in ll if "Montant salarial" in x and "Mt patronal" in x),None)
        if not h:errors.append("Colonnes introuvables page "+str(n));continue
        sx,px=h.index("Montant salarial"),h.index("Mt patronal")
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
rows=[]
for key,p in people.items():
    g=category_totals(p)
    row={"Matricule":p["mat"],"Brut":euro(p["brut"] or 0),"Patronal":euro(p["pat"] or 0)}
    for c in CATEGORIES:row[LABELS[c]]=euro(sum(g[c]))
    row["Écart patronal"]=euro(p["delta"]) if p["delta"] is not None else "Indéterminé"
    rows.append(row)
st.subheader("Synthèse par salarié")
st.dataframe(rows,hide_index=True,use_container_width=True)
if any(p["unknown"] or p["delta"]!=0 for p in people.values()):
    st.warning("Résultat partiel : cotisations inconnues ou écarts de contrôle. Ne pas utiliser comme ventilation validée.")
else:st.success("Contrôle du total patronal : conforme aux bulletins.")
st.subheader("Détail par salarié")
for p in people.values():
    with st.expander("Matricule "+p["mat"],expanded=len(people)==1):
        g=category_totals(p)
        detail=[{"Organisme":LABELS[c],"Salarial":euro(g[c][0]),"Patronal":euro(g[c][1]),"Total":euro(sum(g[c]))} for c in CATEGORIES]
        st.dataframe(detail,hide_index=True,use_container_width=True)
        st.caption("Cotisations classées : "+euro(sum(sum(v) for v in g.values()))+
                   " · Patronal bulletin : "+euro(p["pat"] or 0))
        if p["delta"]!=0:
            st.warning("Écart patronal : "+(euro(p["delta"]) if p["delta"] is not None else "indéterminé"))
        if p["unknown"]:
            st.error(str(len(p["unknown"]))+" rubrique(s) non classée(s).")
            with st.expander("Anomalies techniques"):
                st.dataframe(p["unknown"],hide_index=True,use_container_width=True)

st.caption("URSSAF affichée inclut chômage et AGS. Le tableau des charges reste nécessaire pour valider les rattachements expérimentaux, notamment PRO BTP.")
