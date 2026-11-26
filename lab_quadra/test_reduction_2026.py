"""Cas de comparaison RGDU 2026 : indépendant des codes des dossiers."""
import unittest
from decimal import Decimal
from .reduction_2026 import observed_t1_rate, estimate_standard_split

class TestReduction2026(unittest.TestCase):
    def test_taux_observes(self):
        self.assertEqual(observed_t1_rate([
            {"label":"Complémentaire Tranche 1","base":261289,"pat":15704}]),Decimal("6.01"))
        self.assertEqual(observed_t1_rate([
            {"label":"Complémentaire Tranche 1","base":210710,"pat":11020}]),Decimal("5.23"))
        self.assertIsNone(observed_t1_rate([
            {"label":"Complémentaire Tranche 1","base":261289,"pat":15704},
            {"label":"Complémentaire Tranche 1","base":210710,"pat":11020}]))
    def test_trans2m_octobre_2026(self):
        q802=[70973,81752,81907,81907]
        amounts=[estimate_standard_split(x,Decimal("6.01"),"0.10") for x in q802]
        self.assertEqual([a["retraite"] for a in amounts],[10715,12342,12365,12365])
        self.assertEqual(sum(a["retraite"] for a in amounts),47787)
        self.assertEqual(sum(a["urssaf"]+a["retraite"] for a in amounts),316539)
    def test_axe_dv_octobre_2026(self):
        self.assertEqual(estimate_standard_split(61754,Decimal("6.01"),"0.10")["retraite"],9323)
    def test_elm_not_replaced_by_standard(self):
        self.assertIsNone(estimate_standard_split(77765,Decimal("5.23"),"0.10"))
    def test_unknown_parameters_cannot_be_guessed(self):
        self.assertIsNone(estimate_standard_split(70973,Decimal("6.01")))
        self.assertIsNone(estimate_standard_split(70973,Decimal("5.80"),"0.10"))
        self.assertIsNone(estimate_standard_split(70973,Decimal("6.01"),"0.12"))

if __name__=="__main__":
    unittest.main()
