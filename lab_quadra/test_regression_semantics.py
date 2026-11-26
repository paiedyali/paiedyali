"""Non-regression: contribution labels must not depend on dossier-specific codes.

Run from repository root:
    python -m unittest lab_quadra.test_regression_semantics
"""
import ast
import re
import unicodedata
import unittest
from pathlib import Path

APP=Path(__file__).with_name("app.py")
TREE=ast.parse(APP.read_text(encoding="utf-8"),filename=str(APP))
FUNCTIONS={"normalized_label","non_contribution","semantic_group"}
NODES=[node for node in TREE.body if isinstance(node,ast.FunctionDef)
       and node.name in FUNCTIONS]
SCOPE={"re":re,"unicodedata":unicodedata}
exec(compile(ast.Module(body=NODES,type_ignores=[]),str(APP),"exec"),SCOPE)
semantic_group=SCOPE["semantic_group"]

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

    def test_code_is_never_used_for_collector(self):
        source=APP.read_text(encoding="utf-8")
        self.assertNotIn("MAP.get(code)",source)
        self.assertIn("has_explicit_agirc_reduction",source)
        self.assertIn('x["group"]="URSSAF"',source)

if __name__=="__main__":
    unittest.main()
