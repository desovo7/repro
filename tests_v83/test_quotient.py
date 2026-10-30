from collections import Counter
import unittest
from repro_v83.quotient import CELLS,cell_index


class QuotientTests(unittest.TestCase):
    def test_reachable_nonterminal_quotient_has_96_cells(self):
        self.assertEqual(len(CELLS),96)
        counts=Counter(x[0] for x in CELLS)
        self.assertEqual(counts,{'pick_and_place_simple':12,'pick_clean_then_place_in_recep':15,
            'pick_heat_then_place_in_recep':15,'pick_cool_then_place_in_recep':15,
            'look_at_obj_in_light':18,'pick_two_obj_and_place':21})
        self.assertEqual(set(cell_index(*cell) for cell in CELLS),set(range(96)))

    def test_terminal_is_fixed_and_impossible_nav_is_rejected(self):
        self.assertIsNone(cell_index('pick_two_obj_and_place',4,1,2))
        with self.assertRaises(ValueError):cell_index('pick_and_place_simple',1,0,0)
        with self.assertRaises(ValueError):cell_index('pick_and_place_simple',0,1,3)


if __name__=='__main__':unittest.main()
