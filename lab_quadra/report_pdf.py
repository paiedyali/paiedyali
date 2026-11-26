"""PDF client report for PaieDyali Entreprise laboratory.

Only called once payslip contributions and employer totals reconcile.
All monetary inputs are integer cents.
"""
from io import BytesIO
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, KeepTogether

NAVY = colors.HexColor("#15354A")
TEAL = colors.HexColor("#117A75")
LIGHT = colors.HexColor("#EDF5F5")
GRAY = colors.HexColor("#64748B")
LABELS = ["URSSAF", "Retraite", "Prévoyance", "Mutuelle", "Autres"]
KEYS = ["URSSAF", "RETRAITE", "PREVOYANCE", "MUTUELLE", "AUTRES"]


def fmt(amount):
    return f"{amount/100:,.2f}".replace(",", " ").replace(".", ",") + " €"


def percent(amount, base):
    return (f"{amount * 100 / base:.2f}".replace(".", ",") + " %") if base > 0 else "-"


def make_report(siret, period, employees, estimated=False, company_name=None):
    """Return the PDF bytes; employees include cost, gross, net, groups and contribution totals."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=17*mm,
                            leftMargin=17*mm, topMargin=18*mm, bottomMargin=19*mm,
                            title="PaieDyali Entreprise - Synthèse de paie",
                            author="PaieDyali Entreprise")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="PDTitle", parent=styles["Title"],
                 textColor=NAVY, fontName="Helvetica-Bold", fontSize=19, leading=24,
                 spaceAfter=6))
    styles.add(ParagraphStyle(name="PDSection", parent=styles["Heading2"],
                 textColor=NAVY, fontName="Helvetica-Bold", fontSize=12, leading=15,
                 spaceBefore=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="PDSmall", parent=styles["Normal"],
                 fontName="Helvetica", fontSize=8, leading=11, textColor=GRAY))
    styles.add(ParagraphStyle(name="PDCell", parent=styles["Normal"],
                 fontName="Helvetica", fontSize=8, leading=10))
    styles.add(ParagraphStyle(name="PDCellRight", parent=styles["PDCell"], alignment=TA_RIGHT))
    styles.add(ParagraphStyle(name="PDCellHead", parent=styles["PDCell"],
                 fontName="Helvetica-Bold", textColor=colors.white))
    story = []
    P = lambda value, sty="PDCell": Paragraph(escape(str(value)), styles[sty])
    story += [Paragraph("PaieDyali Entreprise", styles["PDTitle"]),
              Paragraph("Synthèse mensuelle des coûts salariaux", styles["PDSection"])]
    if company_name:
        story.append(Paragraph("Entreprise : "+escape(str(company_name)), styles["Normal"]))
    story += [Paragraph("Période : " + escape(str(period)) + "  |  SIRET : " + escape(str(siret)),
                        styles["PDSmall"]), Spacer(1, 8*mm)]

    cost = sum(e["cost"] for e in employees)
    net = sum(e["net"] for e in employees)
    gross = sum(e["gross"] for e in employees)
    sal = sum(e["sal"] for e in employees)
    pat = sum(e["pat"] for e in employees)
    social = sal + pat

    def table(rows, widths, header=True, total=False):
        t = Table(rows, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
        cmd=[("VALIGN", (0,0), (-1,-1), "MIDDLE"),
             ("LEFTPADDING", (0,0), (-1,-1), 8),
             ("RIGHTPADDING", (0,0), (-1,-1), 7),
             ("TOPPADDING", (0,0), (-1,-1), 6),
             ("BOTTOMPADDING", (0,0), (-1,-1), 6),
             ("LINEBELOW", (0,0), (-1,-1), 0.3, colors.HexColor("#D9E3E7"))]
        if header:
            cmd += [("BACKGROUND", (0,0), (-1,0), NAVY),
                    ("TEXTCOLOR", (0,0), (-1,0), colors.white)]
        if total:
            cmd += [("BACKGROUND", (0,-1), (-1,-1), LIGHT),
                    ("LINEABOVE", (0,-1), (-1,-1), 0.6, TEAL)]
        t.setStyle(TableStyle(cmd))
        return t

    story.append(table([
        [P("Indicateur","PDCellHead"),P("Montant","PDCellHead")],
        [P("Coût global des salariés"),P(fmt(cost),"PDCellRight")],
        [P("Net total à payer"),P(fmt(net),"PDCellRight")],
        [P("Masse salariale brute"),P(fmt(gross),"PDCellRight")],
        [P("Cotisations sociales (salariales + patronales, hors PAS)"),
         P(fmt(social),"PDCellRight")],
    ], [116*mm,52*mm]))

    company_groups={k:[0,0] for k in KEYS}
    for e in employees:
        for k in KEYS:
            a,b=e["groups"].get(k,(0,0))
            company_groups[k][0]+=a
            company_groups[k][1]+=b

    story.append(Paragraph("Répartition des cotisations par organisme",styles["PDSection"]))
    rows=[[P(x,"PDCellHead") for x in ["Organisme","Salariales","Patronales","Total","Part"]]]
    for k,label in zip(KEYS,LABELS):
        a,b=company_groups[k]
        rows.append([P(label),P(fmt(a),"PDCellRight"),P(fmt(b),"PDCellRight"),
                     P(fmt(a+b),"PDCellRight"),P(percent(a+b,social),"PDCellRight")])
    rows.append([P("TOTAL"),P(fmt(sal),"PDCellRight"),P(fmt(pat),"PDCellRight"),
                 P(fmt(social),"PDCellRight"),P(percent(social,social),"PDCellRight")])
    story.append(table(rows,[33*mm,33*mm,33*mm,43*mm,26*mm],total=True))

    story.append(Paragraph("Classement des salariés par coût",styles["PDSection"]))
    rows=[[P(x,"PDCellHead") for x in ["Salarié","Coût employeur","Part du coût"]]]
    for e in sorted(employees,key=lambda x:x["cost"],reverse=True):
        rows.append([P(e["name"]),P(fmt(e["cost"]),"PDCellRight"),
                     P(percent(e["cost"],cost),"PDCellRight")])
    rows.append([P("TOTAL"),P(fmt(cost),"PDCellRight"),P(percent(cost,cost),"PDCellRight")])
    story.append(table(rows,[89*mm,44*mm,35*mm],total=True))

    story.append(Paragraph("Détail par salarié",styles["PDSection"]))
    for e in sorted(employees,key=lambda x:x["cost"],reverse=True):
        subtotal=e["sal"]+e["pat"]
        section=[Paragraph(escape(e["name"]+" - Matricule "+str(e["mat"])),styles["PDSection"])]
        section.append(table([
            [P("Coût employeur","PDCellHead"),P("Part du coût","PDCellHead"),
             P("Brut","PDCellHead"),P("Net à payer","PDCellHead")],
            [P(fmt(e["cost"]),"PDCellRight"),
             P(percent(e["cost"],cost),"PDCellRight"),
             P(fmt(e["gross"]),"PDCellRight"),
             P(fmt(e["net"]),"PDCellRight")],
        ],[42*mm]*4))
        section.append(Spacer(1,3*mm))
        details=[[P(x,"PDCellHead") for x in ["Organisme","Salariales","Patronales","Total","Part"]]]
        for k,label in zip(KEYS,LABELS):
            a,b=e["groups"].get(k,(0,0))
            details.append([P(label),P(fmt(a),"PDCellRight"),P(fmt(b),"PDCellRight"),
                            P(fmt(a+b),"PDCellRight"),P(percent(a+b,subtotal),"PDCellRight")])
        details.append([P("TOTAL"),P(fmt(e["sal"]),"PDCellRight"),
                        P(fmt(e["pat"]),"PDCellRight"),P(fmt(subtotal),"PDCellRight"),
                        P(percent(subtotal,subtotal),"PDCellRight")])
        section.append(table(details,[33*mm,33*mm,33*mm,43*mm,26*mm],total=True))
        section.append(Spacer(1,3*mm))
        story.append(KeepTogether(section))

    story += [Spacer(1, 8*mm),
              Paragraph("Les parts des salariés sont calculées sur le coût global employeur. "
              "Les parts des organismes sont calculées sur les cotisations sociales, hors prélèvement à la source.",
              styles["PDSmall"])]
    if estimated:
        story.append(Paragraph("Ventilation URSSAF / retraite estimée à partir des bulletins. "
            "Document de gestion indicatif, ne constitue pas un état officiel de règlement.",
            styles["PDSmall"]))
    else:
        story.append(Paragraph("Document de synthèse de gestion établi à partir des bulletins "
            "importés. Ne remplace pas les états officiels des organismes.",styles["PDSmall"]))

    def footer(canvas,doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9E3E7"))
        canvas.line(17*mm,15*mm,193*mm,15*mm)
        canvas.setFont("Helvetica",8)
        canvas.setFillColor(GRAY)
        canvas.drawString(17*mm,11*mm,"PaieDyali Entreprise - Rapport de synthèse")
        canvas.drawRightString(193*mm,11*mm,f"Page {doc.page}")
        canvas.restoreState()

    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return buffer.getvalue()
