"""Mandatory real-trace rules, with proposed evidence-based spec corrections.

UT-2's original reference is false in the immutable source. This suite tests
the concrete proposed replacement F22-GiGPO/7 and the complete section-3.3
formula for UT-4/UT-6. A production gate must preserve these documented errata;
passing this suite is not a claim that the erroneous original references pass.
"""
import json
from pathlib import Path
import unittest

from repro_v83.credit import td_credit
from repro_v83.traces import track_trace

ROOT=Path(__file__).resolve().parent
REPLAY=ROOT/'reports/v83_20261030/replayed_traces.jsonl'


class RealTraceRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=[json.loads(line) for line in REPLAY.read_text().splitlines()]
        cls.traces={(r['failure_id'],r['method']):track_trace(r) for r in raw}
        assert len(cls.traces)==45

    def transition(self,fid,method,step):
        trace=self.traces[(fid,method)];tr=trace.transitions[step-1]
        self.assertFalse(tr.feedback_missing,'Mandatory gate uses real CPU replay, not guessed feedback')
        return tr,td_credit(trace.task,tr)[0]

    def test_UT1_exact_target_conflict(self):
        tr,delta=self.transition('F03','RePro',3)
        self.assertEqual((tr.before.p,tr.after.p),(0,-1));self.assertAlmostEqual(delta,-.147)

    def test_UT2_confirmed_real_distractor_release(self):
        tr,delta=self.transition('F22','GiGPO',7)
        self.assertEqual((tr.before.p,tr.after.p),(-1,0));self.assertIsNone(tr.after.held_obj)
        self.assertAlmostEqual(delta,.15)
        # Preserve the failed original-spec reference as a regression check.
        original=self.traces[('F03','RePro')]
        self.assertEqual(original.steps[33]['action'],'go to drawer 1')
        self.assertEqual(original.steps[35]['observation_after'],'Nothing happens.')
        self.assertEqual(original.transitions[35].after.held_obj,'pen 1')

    def test_UT3_unprocessed_goal_placement_and_retrieval(self):
        bad,negative=self.transition('F27','GiGPO',4)
        good,positive=self.transition('F27','GiGPO',14)
        self.assertEqual((bad.before.p,bad.after.p),(1,-1));self.assertLess(negative,0)
        self.assertEqual((good.before.p,good.after.p),(-1,1));self.assertGreater(positive,0)

    def test_UT4_first_lamp_and_repeated_lamp(self):
        first,delta=self.transition('F12','RePro',2)
        self.assertFalse(first.no_op);self.assertTrue(first.first_lamp_activation)
        self.assertIn('desk 1',first.after.lit_lamp_locs);self.assertAlmostEqual(delta,.048)
        for step in [7,13,16,19,26,39,42,48]:
            with self.subTest(step=step):
                repeat,negative=self.transition('F01','RePro',step)
                self.assertTrue(repeat.no_op);self.assertAlmostEqual(negative,-.05)

    def test_UT5_real_terminal_return_to_lamp(self):
        tr,delta=self.transition('F12','GiGPO',10)
        self.assertEqual((tr.before.p,tr.after.p),(1,3));self.assertTrue(tr.after.won)
        self.assertEqual(tr.after.loc,'desk 1');self.assertFalse(tr.no_op);self.assertGreater(delta,0)

    def test_UT6_candidate_return_and_instance_recycling(self):
        good,positive=self.transition('F31','GiGPO',24)
        self.assertEqual((good.before.p,good.after.p),(2,2))
        self.assertEqual(good.after.known_candidates,{'armchair 1'})
        self.assertEqual(good.after.nav_bit,1);self.assertAlmostEqual(positive,.04)
        bad,negative=self.transition('F31','RePro',15)
        self.assertEqual((bad.before.p,bad.after.p),(2,1));self.assertAlmostEqual(negative,-.255)

    def test_all_14_native_winning_steps_have_no_false_penalty(self):
        count=0
        for (fid,method),trace in self.traces.items():
            if method=='GiGPO':
                count+=1;self.assertTrue(trace.success)
                terminal=trace.transitions[-1]
                self.assertFalse(terminal.feedback_missing);self.assertFalse(terminal.no_op)
                self.assertGreater(td_credit(trace.task,terminal)[0],0,fid)
        self.assertEqual(count,14)


if __name__=='__main__':unittest.main()
