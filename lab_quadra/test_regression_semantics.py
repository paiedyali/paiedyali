"""Non-regression: contribution labels must not depend on dossier-specific codes.

Run from repository root:
    python -m unittest lab_quadra.test_regression_semantics
"""
import ast
import re
from decimal import Decimal
import unicodedata
import unittest
from pathlib import Path

APP=Path(__file__).with_name("app.py")
TREE=ast.parse(APP.read_text(encoding="utf-8"),filename=str(APP))
FUNCTIONS={"normalized_label","non_contribution","semantic_group","cents","extract_pas","extract_net_before_tax","extract_net_paye","extract_net_salary_block","health_supplement_is_mutuelle","infer_zero_pas"}
NODES=[node for node in TREE.body if isinstance(node,ast.FunctionDef)
       and node.name in FUNCTIONS]
SCOPE={"re":re,"unicodedata":unicodedata,"Decimal":Decimal,
       "MONEY":re.compile(r"-?\\d+\\.\\d{2,4}")}
exec(compile(ast.Module(body=NODES,type_ignores=[]),str(APP),"exec"),SCOPE)
semantic_group=SCOPE["semantic_group"]
health_supplement_is_mutuelle=SCOPE["health_supplement_is_mutuelle"]
extract_pas=SCOPE["extract_pas"]
infer_zero_pas=SCOPE["infer_zero_pas"]
extract_net_before_tax=SCOPE["extract_net_before_tax"]
extract_net_paye=SCOPE["extract_net_paye"]
extract_net_salary_block=SCOPE["extract_net_salary_block"]

class RegressionSemantics(unittest.TestCase):
    def test_test4_all_22_distinct_labels(self):
        expected={
            "URSSAF": [
                "URSSAF MALADIE-MATERN.-INVAL.","URSSAF ACCIDENT DU TRAVAIL",
                "URSSAF ASS. VIEILLESSE PLAF","URSSAF ASS. VIEILLESSE DEPLAF",
                "FNAL","CONTRIB. SOLIDARITE AUTONOMIE",
                "CONTRIB. DIALOGUE SOCIAL","AF Taux réduit","AF Complément",
                "CSG DEDUCTIBLE","PARTICIPAT° FORMAT° ENT <11",
                "TAXE APPRENTISSAGE","MALADIE TX COMPLEMENT",
                "SOLDE TAXE APPRENTISSAGE",
            ],
            "CHOMAGE":["FONDS DE GARANTIE (AGS)","ASSURANCE CHOMAGE TRA ET TRB"],
            "RETRAITE":["RETRAITE COMPL. AA TU1","CEG TU1",
                        "REDUCTION GLE AGIRC ARRCO"],
            "PREVOYANCE":["FOND ACTION SOCIALE",
                          "PREVOYANCE RESTAURAT RAPIDE TA"],
            "REDUCTION_A_QUALIFIER":["REDUCTION GENERALE"],
        }
        self.assertEqual(sum(map(len,expected.values())),22)
        for group, labels in expected.items():
            for label in labels:
                with self.subTest(label=label):
                    self.assertEqual(semantic_group(label),group)

    def test_convention_module_and_negative_adjustments(self):
        expected={
            "Complémentaire Incap.Inval.Déc":"PREVOYANCE",
            "Complémentaire Tranche 2":"RETRAITE",
            "APEC":"RETRAITE",
            "EPS8 PREVOYANCE CADRE TR B":"PREVOYANCE",
            "EPS7 PREVOYANCE CADRE TR A":"PREVOYANCE",
            "MUTUELLE CADRE ISOLE":"MUTUELLE",
            "Cotisations statutaires ou CC":"AUTRES",
            "EXO., ECRET. ET ALLEG. COTIS":"REDUCTION_MIXTE",
            "Indemnité repas hors resto":"EXCLUDE",
            "SP08 INDEMNITE REPAS HORS RESTO":"EXCLUDE",
        }
        for label,group in expected.items():
            with self.subTest(label=label):
                self.assertEqual(semantic_group(label),group)

    def test_btp_conventional_and_overtime_labels(self):
        labels={
            "BTP-PREV CHIRURGIE CADRE TA":"PREVOYANCE",
            "BTP-PREVOYANCE CADRE OBLIG TA":"PREVOYANCE",
            "BTP-PREVOYANCE OBLIGAT OUVRIER":"PREVOYANCE",
            "BTP-PREV HOSP CHIR OUVRIER":"PREVOYANCE",
            "BTP-CONSTRUCTYS HT Tx convent.":"PREVOYANCE",
            "BTP-CONSTRUCTYS TVA Tx Conv":"PREVOYANCE",
            "BTP-APNAB":"PREVOYANCE",
            "BTP - TAXE CCCA":"PREVOYANCE",
            "BTP CONSTRUCTYS TX LEG":"URSSAF",
            "BTP - CSG DEDUCTIBLE OUVRIER":"URSSAF",
            "PARTICIP FORM ENT <11 BTP":"URSSAF",
            "TAXE APPRENTISSAGE BTP":"URSSAF",
            "REDUCTION SALARIALE HS/HC 2019":"URSSAF",
            "REDUCT HEURES SUPPL. P.P.<= 20":"URSSAF",
            "APEC TR A":"RETRAITE",
        }
        for label,group in labels.items():
            with self.subTest(label=label):
                self.assertEqual(semantic_group(label),group)

    def test_inelit_november_2026_regression(self):
        # Montants tirés des bulletins INELIT, les codes E_85/E_86/E_D2
        # ne servent jamais à déterminer l'organisme.
        cases=[
            ("RENTE EDUCATION TRA","PREVOYANCE",181,181),
            ("RENTE EDUCATION TRB","PREVOYANCE",12,12),
            ("ADESATT","AUTRES",0,48),
            ("ADESATT","AUTRES",0,83),
            ("PARTICIP FORMATION CDD","URSSAF",0,2410),
        ]
        for label,group,sal,pat in cases:
            with self.subTest(label=label,pat=pat):
                self.assertEqual(semantic_group(label),group)
        # Les deux lignes non reconnues du matricule 4 expliquent
        # exactement son ecart patronal initial de -0,95 EUR.
        self.assertEqual(12+83,95)
        self.assertEqual(181+48+2410,2639)

    def test_quadrapayroll_two_pages_blank_zero_pas(self):
        # Capture du matricule 9 : la colonne Montant fiscal est vide,
        # le cumul annuel indique 0,00. Le net imprimé est identique
        # au net avant impôt de 3 492,69 EUR sur la seconde page.
        self.assertTrue(infer_zero_pas(None,True,349269,349269,"bulletin"))
        self.assertFalse(infer_zero_pas(None,False,349269,349269,"bulletin"))
        self.assertFalse(infer_zero_pas(None,True,349269,349270,"bulletin"))
        self.assertFalse(infer_zero_pas(None,True,None,349269,None))
        self.assertFalse(infer_zero_pas(None,True,349269,None,"bulletin"))
        self.assertFalse(infer_zero_pas(821,True,349269,349269,"bulletin"))
        # Zero n'est jamais supposé à partir du seul cumul annuel.
        self.assertIsNone(extract_pas(
            "Impôt sur le revenu prélevé à la source                            0.00"))

    def test_pas_dgfip_extraction_for_references(self):
        # 4 montants: base, taux, retenue, cumul. Ne pas prendre le cumul.
        self.assertEqual(extract_pas(
            "Impôt sur le revenu prélevé à la source 2626.27 8.10 212.73 351.92"),21273)
        self.assertEqual(extract_pas(
            "Impôt sur le revenu prélevé à la source 1697.72 0.60 10.19 Paiement"),1019)
        self.assertEqual(extract_pas(
            "Impôt sur le revenu prélevé à la source 1630.47 0.00 0.00 0.00"),0)
        self.assertEqual(extract_pas("Impôt sur le revenu prélevé à la source"),None)
        self.assertEqual(extract_pas(
            "Impôt sur le revenu prélevé à la source 1638.26 0.50 8.19 8.19"),819)
        self.assertEqual(extract_pas(
            "Impôt sur le revenu prélevé à la source 851.01 0.00 0.00 0.00"),0)

    def test_les_delices_2026_informative_remuneration_line(self):
        # Boulangerie Q100-Q802: this is an explanatory net-pay line,
        # not an extra URSSAF / unemployment contribution.
        informational="dont évolution de la rémunération liée à la suppression des cotisations salariales chômage et maladie"
        self.assertEqual(semantic_group(informational),"EXCLUDE")
        screenshot_differences=[3010,803,2755,2755,1532,2755,1204,2294]
        self.assertEqual(sum(screenshot_differences),17108)
        source=APP.read_text(encoding="utf-8")
        self.assertIn('if normalized_label(line).startswith("DONT EVOLUTION DE LA REMUNERATION"):',source)

    def test_net_reading_old_quadra_and_clarified_payslips(self):
        self.assertEqual(extract_net_paye("NET PAYE    1 000.00 Euros"),None)
        self.assertEqual(extract_net_paye("NET PAYE    1884.06 Euros"),188406)
        self.assertEqual(extract_net_paye("NET PAYE    0.00     Euros"),0)
        self.assertEqual(extract_net_before_tax("NET A PAYER AVANT IMPOT SUR LE REVENU   3600.84    Total"),360084)
        self.assertEqual(extract_net_before_tax("NET A PAYER AVANT IMPOT SUR LE REVENU 2072.94"),207294)
        self.assertEqual(360084-21273,338811)  # INELIT
        self.assertEqual(207294-1019,206275)  # ELM
        self.assertEqual(218856-3886,214970) # Juin
        self.assertEqual(157980-819,157161)  # BTP
        self.assertEqual(extract_net_salary_block(
            ["Net à payer au salarié (En Euros)",
             "Convention collective nationale des bureaux d'études techniques",
             "cabinets d'ingénieurs-conseils et sociétés de conseils",
             "           2110.90",
             "A défaut de Convention Collective"],0),211090)
        self.assertEqual(extract_net_salary_block(
            ["Net à payer au salarié (En Euros)","A défaut de Convention Collective"],0),None)

    def test_axe_dv_october_2026(self):
        """Les codes changent; le libellé et la nature déterminent le collecteur."""
        cases={
            "VERS SERVICES MOBILITE ENT >11":"URSSAF",
            "REDUCTION HS PP > 20":"URSSAF",
            "FORFAIT SOCIAL":"URSSAF",
        }
        for label, group in cases.items():
            with self.subTest(label=label):
                self.assertEqual(semantic_group(label),group)
        # Écarts patronaux des deux bulletins AXE DV avant correction.
        self.assertEqual(10239-867+304,9676)
        self.assertEqual(4066+121,4187)
        self.assertEqual(9676+4187,13863)

    def test_axe_dv_full_bundle_allowances_are_not_cotisations(self):
        # Le meme matricule doit donner la meme ventilation pour un PDF
        # de 3 pages ou un PDF complet de 25 pages. Les primes SP07/SP08
        # suivent le net imposable et ne sont jamais des cotisations.
        for label in (
            "SP07 Prime de panier",
            "SP08 Prime de Salissure",
            "Prime de panier",
            "Prime de salissure",
            "INDEMNITE REPAS HORS RESTO",
        ):
            with self.subTest(label=label):
                self.assertEqual(semantic_group(label,"Q802"),"EXCLUDE")
        # Ne pas classer tous les SP aveuglement: seul le sens compte.
        self.assertIsNone(semantic_group("SP10 libelle conventionnel inconnu","Q802"))
        # Les deux montants de 243 et 36,21 sont inclus dans le net/cout,
        # pas dans le total des cotisations.
        self.assertEqual(24300+3621,27921)

    def test_axe_dv_wage_garnishment_not_a_contribution(self):
        # Lignes Q801 de saisie-arrêt dans l'import AXE DV complet.
        # Ne jamais compter ces retenues parmi les cotisations sociales,
        # ni soustraire à nouveau du net déjà imprimé.
        for label in ("Saisie arrêt", "SAISIE ARRET",
                      "Saisie sur salaire", "Saisie sur rémunération"):
            with self.subTest(label=label):
                self.assertEqual(semantic_group(label, "Q801"), "EXCLUDE")
        self.assertEqual(33890+25186+14383,73459)
        # Le net du matricule 242 (capture technique) reste déterminé
        # explicitement par net avant impôt - PAS ; sans autre retraitement.
        self.assertEqual(992113-160373,831740)

    def test_mutuelle_visible_without_mixing_with_reduction(self):
        # AXE DV : l'organisme mutuelle s'identifie au libellé
        # (non au code E004), même si Q802 est présent ailleurs.
        self.assertEqual(semantic_group("Frais de santé"),"MUTUELLE")
        self.assertEqual(semantic_group("MUTUELLE +opt2"),"MUTUELLE")
        # "Complémentaire Santé" n'est pas un collecteur fiable :
        # dans ELM une telle ligne relève du bloc prévoyance.
        self.assertIsNone(semantic_group("Complémentaire Santé","Q100"))
        source=APP.read_text(encoding="utf-8")
        self.assertIn('st.subheader("Mutuelle identifiée dans les bulletins")',source)
        self.assertIn('if x["group"]=="MUTUELLE"',source)

    def test_q_long_bulletin_health_subtotal_not_counted_twice(self):
        # Dans un bulletin avec sous-rubriques Q, ce sous-total figure
        # apres les cotisations. Il ne doit PAS s'additionner une 2e fois.
        for label in (
            "Cotis. Retraite/Prév./F.santé",
            "Cotis. Retraite/Prev./F.sante",
            "Cotis. Retraite/Prév./F.santé 31.02",
        ):
            with self.subTest(label=label):
                self.assertEqual(semantic_group(label,"Q801"),"EXCLUDE")
        # AXE DV complet : 8 bulletins affichent une ligne Q100 identique,
        # sans qu'on puisse compter sur la présence du taux dans le texte.
        for _ in range(8):
            self.assertTrue(health_supplement_is_mutuelle(
                "Complémentaire Santé","Complémentaire Santé",3102,3102))
        self.assertEqual(8*(3102+3102),49632)
        self.assertEqual(154548-49632,104916)
        self.assertEqual(4058551-49632,4008919)
        # Q100: 50/50 a taux 50% = mutuelle a somme fixe.
        self.assertTrue(health_supplement_is_mutuelle(
            "Complémentaire Santé","Complémentaire Santé  62.03  50.0000",3102,3102))
        # Q100 ELM: cotisation assise sur brut, taux 1,24 %,
        # 26,13 salarié + 39,19 patronal -> prévoyance.
        self.assertFalse(health_supplement_is_mutuelle(
            "Complémentaire Santé","Complémentaire Santé  2107.10  1.2400",2613,3919))
        self.assertFalse(health_supplement_is_mutuelle(
            "Complémentaire Santé","Complémentaire Santé  2400.54  0.4800",1152,1728))
        # AXE: exactement 496,32 de difference en salariales entre
        # l'ancienne synthese et le tableau des charges; ne pas coder
        # un correctif forfaitaire sur cette valeur.
        self.assertEqual(123828-(49380+24816),49632)
        self.assertEqual(80352-(55536+24816),0)
        self.assertEqual(4058551-4008919,49632)

    def test_code_is_never_used_for_collector(self):
        source=APP.read_text(encoding="utf-8")
        self.assertNotIn("MAP.get(code)",source)
        self.assertIn("has_explicit_agirc_reduction",source)
        self.assertIn('x["group"]="URSSAF"',source)

if __name__=="__main__":
    unittest.main()
