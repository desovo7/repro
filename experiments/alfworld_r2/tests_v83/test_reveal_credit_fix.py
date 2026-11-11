"""Semantic regressions for first reveals and exhaustive AF transition logs."""
import copy
import json
from pathlib import Path
import unittest

from repro_v83.tracker import Task,TrajectoryStateTracker,is_search
from repro_v83.credit import td_credit,centered_auxiliary
from repro_v83.traces import TrackedTrace,track_trace
from repro_v83.value_table import QuotientValueTable


class RevealCreditTests(unittest.TestCase):
    def tracker(self,family='simple'):
        t=TrajectoryStateTracker(Task(family,'pencil','desk'))
        t.reset('You are in the middle of a room.')
        return t

    def closed(self,t,host='drawer 1'):
        return t.step('go to '+host,f'You arrive at {host}. The {host} is closed.')

    def opened(self,t,host='drawer 1',contents='nothing',**kwargs):
        return t.step('open '+host,f'You open the {host}. The {host} is open. In it, you see {contents}.',**kwargs)

    def test_first_open_reveals_target_and_empty_contents(self):
        for host in ['drawer 1','cabinet 1','fridge 1','safe 1']:
            for contents in ['nothing','a pencil 1']:
                with self.subTest(host=host,contents=contents):
                    t=self.tracker();arrival=self.closed(t,host)
                    tr=self.opened(t,host,contents)
                    self.assertIn(host,arrival.after.visited_receps_subgoal)
                    self.assertNotIn(host,arrival.after.revealed_receps)
                    self.assertTrue(tr.first_contents_reveal)
                    self.assertEqual((tr.before.p,tr.after.p,tr.after.nav_bit),(0,0,1))
                    self.assertAlmostEqual(td_credit(t.task,tr)[0],.05)
                    snapshot=QuotientValueTable().snapshot(1)
                    values=(snapshot.value_for_state(t.task,tr.before,2),snapshot.value_for_state(t.task,tr.after,3))
                    self.assertAlmostEqual(td_credit(t.task,tr,values=values)[0],-.001)

    def test_close_reopen_does_not_create_a_second_reveal(self):
        t=self.tracker();self.closed(t);self.opened(t)
        t.step('close drawer 1','You close the drawer 1.')
        tr=self.opened(t)
        self.assertFalse(tr.first_contents_reveal)
        self.assertEqual(tr.after.nav_bit,0)
        self.assertAlmostEqual(td_credit(t.task,tr)[0],-.05)

    def test_contents_seen_on_arrival_are_not_new_on_reopening(self):
        t=self.tracker()
        t.step('go to drawer 1','You arrive at drawer 1. The drawer 1 is open. In it, you see nothing.')
        t.step('close drawer 1','You close the drawer 1.')
        self.assertFalse(self.opened(t).first_contents_reveal)

    def test_reveal_memory_survives_subgoal_change_but_resets_per_episode(self):
        t=self.tracker();self.closed(t);self.opened(t)
        t.step('take pen 1 from drawer 1','You pick up the pen 1 from the drawer 1.')
        t.step('move pen 1 to drawer 1','You move the pen 1 to the drawer 1.')
        t.step('close drawer 1','You close the drawer 1.')
        self.assertFalse(self.opened(t).first_contents_reveal)
        t.reset('You are in the middle of a room.');self.closed(t)
        self.assertTrue(self.opened(t).first_contents_reveal)

    def test_first_reveal_also_covers_second_object_search(self):
        t=self.tracker('pick_two');self.closed(t,'drawer 2')
        t.step('take pencil 1 from drawer 2','You pick up the pencil 1 from the drawer 2.')
        t.step('move pencil 1 to desk 1','You move the pencil 1 to the desk 1.')
        self.closed(t)
        tr=self.opened(t,contents='a pencil 2')
        self.assertEqual((tr.before.p,tr.after.p),(2,2))
        self.assertEqual(tr.after.nav_bit,1)
        self.assertAlmostEqual(td_credit(t.task,tr)[0],.04)

    def test_invalid_or_unconfirmed_open_is_not_a_reveal_event(self):
        for action,text,valid in [
            ('open drawer 1','Nothing happens.',True),
            ('open drawer 1',None,True),
            ('open drawer 2','You open the drawer 1. In it, you see nothing.',True),
            ('open drawer 1','You open the drawer 1. In it, you see nothing.',False),
            ('open drawer 1','You open the drawer 1.',True),
        ]:
            with self.subTest(action=action,text=text,valid=valid):
                t=self.tracker();self.closed(t)
                tr=t.step(action,text,format_valid=valid)
                self.assertFalse(tr.first_contents_reveal)
                self.assertEqual(tr.after.nav_bit,0)

    def test_first_lamp_uses_one_reward_without_navigation_double_count(self):
        t=self.tracker('look_at')
        t.step('go to desk 1','You arrive at desk 1. On the desk 1, you see a desklamp 1.')
        tr=t.step('use desklamp 1','You turn on the desklamp 1.')
        self.assertTrue(tr.first_lamp_activation)
        self.assertEqual(tr.after.nav_bit,1)
        self.assertAlmostEqual(td_credit(t.task,tr)[0],.098)
        snapshot=QuotientValueTable().snapshot(1)
        pair=(snapshot.value_for_state(t.task,tr.before,2),snapshot.value_for_state(t.task,tr.after,3))
        self.assertAlmostEqual(td_credit(t.task,tr,values=pair)[0],.099)
        repeat=t.step('use desklamp 1','You turn on the desklamp 1.')
        self.assertTrue(repeat.no_op);self.assertFalse(repeat.first_lamp_activation)
        self.assertEqual(td_credit(t.task,repeat),(-.05,0.))

    def test_lamp_off_on_cycle_is_not_another_first_activation(self):
        t=self.tracker('look_at')
        t.step('go to desk 1','You arrive at desk 1. On the desk 1, you see a desklamp 1.')
        t.step('use desklamp 1','You turn on the desklamp 1.')
        off=t.step('use desklamp 1','You turn off the desklamp 1.')
        on=t.step('use desklamp 1','You turn on the desklamp 1.')
        self.assertFalse(on.first_lamp_activation)
        self.assertEqual(on.after.nav_bit,0)
        self.assertIn('desk 1',on.after.lit_lamp_locs)
        self.assertLess(td_credit(t.task,off)[0]+td_credit(t.task,on)[0],0)
        self.assertAlmostEqual(td_credit(t.task,on,values=(0.,0.))[0],0.)

    def test_real_historical_first_open_cases_are_all_recognized(self):
        path=Path(__file__).resolve().parents[1]/'reports/v83_20261030/replayed_traces.jsonl'
        count=0
        for line in path.read_text().splitlines():
            trace=track_trace(json.loads(line))
            for step,tr in zip(trace.steps,trace.transitions):
                if (step['action'].startswith('open ') and not tr.no_op
                    and tr.before.p==tr.after.p and is_search(trace.task,tr.before.p)):
                    count+=1
                    self.assertTrue(tr.first_contents_reveal)
                    self.assertEqual(tr.after.nav_bit,1)
                    self.assertGreater(td_credit(trace.task,tr)[0],0)
        self.assertEqual(count,48)


class TransitionDiagnosticTests(unittest.TestCase):
    def traces(self):
        t=TrajectoryStateTracker(Task('simple','pencil','desk'))
        t.reset('You arrive at desk 1. On the desk 1, you see a pen 1.')
        transitions=[
            t.step('take pen 1 from desk 1','You pick up the pen 1 from the desk 1.'),
            t.step('go to shelf 1','You arrive at shelf 1.'),
            t.step('inventory','You are carrying a pen 1.'),
            t.step('move pen 1 to shelf 1','You move the pen 1 to the shelf 1.'),
            t.step('go to drawer 1','You arrive at drawer 1. The drawer 1 is closed.'),
            t.step('open drawer 1','You open the drawer 1. In it, you see nothing.'),
        ]
        return [TrackedTrace(t.task,'task','all_failed',str(i),False,True,
            [dict(action_uid=f'{i}:{j}',step_id=j) for j in range(6)],transitions,{}) for i in range(8)]

    def test_exhaustive_totals_include_noops_and_credit_stages(self):
        traces=self.traces();r=centered_auxiliary(traces);d=r['diagnostics']
        self.assertEqual(d['af_actions'],48)
        for relation,total in [('F_to_D',8),('D_to_D',16),('D_to_F',8),('F_to_F',16)]:
            self.assertEqual(d[f'transition_{relation}_total'],total)
        self.assertEqual(d['transition_D_to_D_noop_actions'],8)
        self.assertEqual(sum(d[f'transition_{k}_nonzero_centered_aux'] for k in ['D_to_D','D_to_F','F_to_D','F_to_F']),d['nonzero_centered_actions'])
        for prefix,values in [('td',r['deltas']),('raw_aux',r['raw']),('centered_aux',r['centered'])]:
            self.assertEqual(d[f'transition_F_to_D_nonzero_{prefix}'],sum(values[f'{i}:0']!=0 for i in range(8)))
            self.assertEqual(d[f'transition_D_to_D_nonzero_{prefix}'],sum(values[f'{i}:{j}']!=0 for i in range(8) for j in [1,2]))

    def test_disabled_or_non_af_groups_do_not_enter_af_counts(self):
        traces=self.traces()
        for disabled in [True,False]:
            if not disabled:traces[0].success=True
            d=centered_auxiliary(traces,enabled=not disabled)['diagnostics']
            self.assertEqual(d['af_actions'],0)
            self.assertTrue(all(v==0 for k,v in d.items() if k.startswith('transition_')))


if __name__=='__main__':unittest.main()
