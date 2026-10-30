import math
from types import SimpleNamespace
import unittest

from repro_v83.credit import (CreditConfig,td_credit,trajectory_gae,budget_bucket,
                              centered_auxiliary,weighted_std)
from repro_v83.tracker import Task,TrajectoryStateTracker


class CreditTests(unittest.TestCase):
    def test_conflict_then_real_recovery_have_spec_signs(self):
        task=Task('simple','pencil','desk');tracker=TrajectoryStateTracker(task)
        tracker.reset('You arrive at desk 1. On the desk 1, you see a pen 1.')
        bad=tracker.step('take pen 1 from desk 1','You pick up the pen 1 from the desk 1.')
        recovered=tracker.step('move pen 1 to desk 1','You move the pen 1 to the desk 1.')
        self.assertAlmostEqual(td_credit(task,bad)[0],-.147)
        self.assertAlmostEqual(td_credit(task,recovered)[0],.15)

    def test_noop_gae_does_not_receive_future_credit(self):
        task=Task('simple','pencil','desk');tracker=TrajectoryStateTracker(task)
        tracker.reset('You arrive at shelf 1. On the shelf 1, you see a pencil 1.')
        noop=tracker.step('examine shelf 1',"There's nothing special about shelf 1.")
        take=tracker.step('take pencil 1 from shelf 1','You pick up the pencil 1 from the shelf 1.')
        gae,deltas,decays=trajectory_gae(task,[noop,take])
        self.assertEqual(decays,[0,.5]);self.assertAlmostEqual(gae[0],-.05)
        self.assertGreater(gae[1],0)

    def test_missing_feedback_is_not_silently_scored(self):
        task=Task('simple','pencil','desk');tracker=TrajectoryStateTracker(task)
        tracker.reset('You arrive at desk 1.')
        tr=tracker.step('look',None)
        with self.assertRaises(ValueError):td_credit(task,tr)

    def test_bucket_boundaries_and_token_std(self):
        self.assertEqual([budget_bucket(x) for x in (0,9,10,24,25,49)],[0,0,1,1,2,2])
        # Expanded tokens [-1,-1,-1,1] have mean -.5 and population variance .75.
        self.assertAlmostEqual(weighted_std([-1,1],[3,1]),math.sqrt(.75))
        self.assertEqual(weighted_std([0],[0]),0)

    def test_group_centering_af_gate_and_uid_fail_closed(self):
        task=Task('simple','pencil','desk')
        def trace(i,won=False):
            tracker=TrajectoryStateTracker(task)
            tracker.reset('You arrive at shelf 1. On the shelf 1, you see a pencil 1.')
            tr=tracker.step('take pencil 1 from shelf 1' if i%2 else 'inventory',
                'You pick up the pencil 1 from the shelf 1.' if i%2 else 'You are not carrying anything.',won=won)
            return SimpleNamespace(task=task,task_uid='same',group_uid='group',trajectory_uid=f't{i}',
                                   success=won,complete=True,steps=[{'action_uid':f'a{i}','step_id':0}],transitions=[tr])
        traces=[trace(i) for i in range(8)]
        result=centered_auxiliary(traces)
        self.assertTrue(result['af']['group'])
        self.assertAlmostEqual(sum(result['centered'].values()),0)
        traces[-1]=trace(7,True)
        result=centered_auxiliary(traces)
        self.assertFalse(result['af']['group'])
        self.assertTrue(all(x==0 for x in result['centered'].values()))
        with self.assertRaises(ValueError):centered_auxiliary(traces[:-1])
        with self.assertRaises(ValueError):centered_auxiliary([traces[0]]*8)


if __name__=='__main__':unittest.main()
