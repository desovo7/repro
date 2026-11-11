"""v8.3.1 exact-formula and transactional Tier 2 integration checks (CPU)."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from repro_v83.anchors import anchor_keys
from repro_v83.credit import center_within_bucket,td_credit,trajectory_gae,weighted_std
from repro_v83.runtime import RuntimeV83
from repro_v83.traces import TrackedTrace,track_trace
from repro_v83.tracker import Task,TrajectoryStateTracker
from repro_v83.value_table import QuotientValueTable

ROOT=Path(__file__).resolve().parent


class Tier2FormulaTests(unittest.TestCase):
    def test_no_navigation_double_count_and_timeout_boundaries(self):
        task=Task('simple','pencil','desk');tracker=TrajectoryStateTracker(task)
        tracker.reset('You arrive at shelf 1.')
        transition=tracker.step('go to shelf 2','You arrive at shelf 2.')
        self.assertTrue(transition.after.nav_bit)
        # Exact table TD: .98*.3-.2=.094; no extra +.05 navigation term.
        self.assertAlmostEqual(td_credit(task,transition,values=(.2,.3))[0],.094)
        for horizon in (10,25,50):
            _,deltas,_=trajectory_gae(task,[transition]*horizon,values=[.2]*(horizon+1))
            self.assertAlmostEqual(deltas[-1],-.2)
            self.assertAlmostEqual(deltas[-2],-.004)

    def test_noop_truncation_and_lamp_bonus(self):
        task=Task('look_at','pencil');tracker=TrajectoryStateTracker(task)
        tracker.reset('You arrive at desk 1. On the desk 1, you see a desklamp 1.')
        first=tracker.step('use desklamp 1','You turn on the desklamp 1.')
        self.assertTrue(first.first_lamp_activation)
        self.assertAlmostEqual(td_credit(task,first,values=(.1,.1))[0],.098)
        noop=tracker.step('use desklamp 1','Nothing happens.')
        gae,delta,decay=trajectory_gae(task,[noop,first],values=[.1,.1,.1])
        self.assertEqual(decay[0],0);self.assertEqual(delta[0],-.05);self.assertEqual(gae[0],-.05)

    def test_UT8_bucket_mean_offset_invariance_and_variance_gate(self):
        for values in ([1.0,-2.0,4.0,0.0]*8,[-.05]*8,[1.0],[1,1+1e-7]):
            result=center_within_bucket(values)
            shifted=center_within_bucket([v+17.25 for v in values])
            self.assertLess(abs(math.fsum(result)/len(result)),1e-12)
            for a,b in zip(result,shifted):self.assertAlmostEqual(a,b,places=12)
        self.assertEqual(center_within_bucket([1,1+1e-7]),[0.,0.])
        with self.assertRaises(ValueError):center_within_bucket([float('nan')])

    def test_exact_formula_does_not_claim_universal_sign_guarantee(self):
        task=Task('simple','pencil','desk');tracker=TrajectoryStateTracker(task)
        tracker.reset('You arrive at shelf 1. On the shelf 1, you see a pencil 1.')
        progress=tracker.step('take pencil 1 from shelf 1','You pick up the pencil 1 from the shelf 1.')
        # Separate budget buckets can assign a lower value to the later state.
        self.assertLess(td_credit(task,progress,values=(.9,.1))[0],0)
        self.assertEqual((progress.before.p,progress.after.p),(0,1))
        # A .02 navigation floor also fails when the preceding value exceeds .98.
        search=progress._replace(before=progress.before,after=progress.before)
        self.assertLess(td_credit(task,search,values=(1.01,1.03))[0],0)


class Tier2RealTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw=[json.loads(line) for line in (ROOT/'reports/v83_20261030/replayed_traces.jsonl').read_text().splitlines()]
        cls.traces={(r['failure_id'],r['method']):track_trace(r) for r in raw}
        cls.snapshot=QuotientValueTable().snapshot(1)

    def delta(self,fid,method,step):
        trace=self.traces[(fid,method)]
        _,deltas,_=trajectory_gae(trace.task,trace.transitions,values=self.snapshot.values_for_trace(trace))
        return deltas[step-1]

    def test_UT1_UT2_UT3_conflict_and_recovery(self):
        self.assertLess(self.delta('F03','RePro',3),0)
        self.assertGreater(self.delta('F22','GiGPO',7),0)  # recorded UT-2 source erratum
        self.assertLess(self.delta('F27','GiGPO',4),0)
        self.assertGreater(self.delta('F27','GiGPO',14),0)

    def test_UT4_UT5_UT6_navigation_lamp_and_recycling(self):
        self.assertGreater(self.delta('F12','RePro',2),0)
        for step in (7,13,16,19,26,39,42,48):self.assertEqual(self.delta('F01','RePro',step),-.05)
        self.assertGreater(self.delta('F12','GiGPO',10),0)
        self.assertGreater(self.delta('F31','GiGPO',24),0)
        self.assertLess(self.delta('F31','RePro',15),0)

    def test_all_14_terminal_steps_cold_start(self):
        count=0
        for trace in self.traces.values():
            if trace.success:
                count+=1;self.assertFalse(trace.transitions[-1].no_op)
                values=self.snapshot.values_for_trace(trace)
                self.assertEqual(values[-1],1.)
                _,deltas,_=trajectory_gae(trace.task,trace.transitions,values=values)
                self.assertGreater(deltas[-1],0,trace.trajectory_uid)
        self.assertEqual(count,14)


def group_traces(iteration):
    traces=[];task=Task('simple','pencil','desk')
    for group in ('mixed','af'):
        for j in range(8):
            success=group=='mixed' and j==0
            tracker=TrajectoryStateTracker(task)
            tracker.reset('You arrive at shelf 1. On the shelf 1, you see a pencil 1.')
            uid=f'i{iteration}-{group}-{j}'
            actions=[('take pencil 1 from shelf 1','You pick up the pencil 1 from the shelf 1.',False),
                     ('go to desk 1','You arrive at desk 1.',False)]
            if success:actions.append(('move pencil 1 to desk 1','You move the pencil 1 to the desk 1.',True))
            else:actions.extend([('inventory','You are carrying a pencil 1.',False)]*48)
            transitions=[];steps=[]
            for i,(action,after,won) in enumerate(actions):
                transition=tracker.step(action,after,won=won)
                transitions.append(transition)
                steps.append(dict(action_uid=f'{uid}:{i}',step_id=i,action=action,
                    observation_before=transition.before.observation,observation_after=after,won=won,format_valid=True))
            source=dict(run_id='v831-cpu',trajectory_uid=uid,group_uid=f'i{iteration}-{group}',task_uid='same-task',
                success=success,iteration_id=iteration,phase='train',split='train',steps=steps)
            traces.append(TrackedTrace(task,'same-task',source['group_uid'],uid,success,True,steps,transitions,source))
    return traces


class Tier2RuntimeTests(unittest.TestCase):
    def setUp(self):self.directory=tempfile.TemporaryDirectory()
    def tearDown(self):self.directory.cleanup()

    def config(self):
        return dict(env=dict(max_steps=50,rollout=dict(n=8)),algorithm=dict(gigpo=dict(enable_similarity=False)),
            repro_v83=dict(run_id='v831-cpu',config_hash='frozen',model_contract_hash='model',
                output_dir=self.directory.name,enable_tier1a=True,enable_tier1b=True,enable_tier2=True,
                tier2_pool_mode='mixed_only'))

    def collected(self,runtime,iteration):
        runtime.start_iteration(iteration);traces=group_traces(iteration)
        with patch('repro_v83.runtime.track_trace',side_effect=traces):
            runtime.after_collection([t.source for t in traces])
        return traces

    def injected(self,runtime):
        import torch
        ids=list(runtime.action_groups)
        # Model actual-response-token cardinality and repeated/reordered originals.
        ids=list(reversed(ids))+ids[:5]
        mask=torch.tensor([[1.,float(i%2),0.] for i in range(len(ids))])
        batch=SimpleNamespace(batch={'advantages':torch.ones_like(mask),'response_mask':mask},
                              non_tensor_batch={'repro_action_uid':ids},meta_info={})
        runtime.inject(batch)
        for i,uid in enumerate(ids):
            if not runtime.credits['af'][runtime.action_groups[uid]]:
                self.assertTrue(torch.equal(batch.batch['advantages'][i],torch.ones(3)))
        self.assertTrue(torch.equal(batch.batch['advantages'][:,2],torch.ones(len(ids))))
        return batch

    def output(self,runtime,valid=True):
        return SimpleNamespace(meta_info={'metrics':{},'repro_external_audit':dict(
            run_id=runtime.run_id,iteration_id=runtime.iteration_id,snapshot_id=runtime.snapshot_id,
            actual_steps_min=1,actual_steps_max=1,expected_steps_min=1,expected_steps_max=1,
            all_ranks_finite=valid,world_size=2)})

    def test_donors_commit_only_after_verified_update_and_eval_isolation(self):
        runtime=RuntimeV83(self.config());self.collected(runtime,1)
        before=copy.deepcopy(runtime.value_table.state_dict())
        runtime.record_evaluation([{'success':True}],runtime.context(1))
        self.assertEqual(runtime.value_table.state_dict(),before)
        self.injected(runtime)
        with self.assertRaises(RuntimeError):runtime.commit(self.output(runtime,False))
        self.assertEqual(runtime.value_table.state_dict(),before)
        runtime.commit(self.output(runtime))
        self.assertEqual(runtime.value_table.state_dict()['last_committed_iteration'],1)
        with self.assertRaises(RuntimeError):runtime.commit(self.output(runtime))
        self.assertTrue((Path(self.directory.name)/'diagnostics/iteration_1.json').exists())
        self.assertIn('singleton_ratio',runtime.anchor_stats['selected'])

    def test_foreign_iteration_and_evaluation_collection_rejected_before_injection(self):
        for changes in ({'split':'valid_seen'},{'iteration_id':2},{'run_id':'another-run'}):
            runtime=RuntimeV83(self.config());runtime.start_iteration(1)
            source=[t.source for t in group_traces(1)];source[0].update(changes)
            with self.assertRaises(ValueError):runtime.after_collection(source)
            self.assertEqual(runtime.phase,'collecting')
            self.assertEqual(runtime.value_table.last_committed_iteration,0)

    def test_history_roundtrip_matches_uninterrupted_next_iteration(self):
        runtime=RuntimeV83(self.config())
        for i in range(1,6):
            self.collected(runtime,i);self.injected(runtime);runtime.commit(self.output(runtime))
        folder=Path(self.directory.name)/'global_step_5';folder.mkdir()
        cursors={'train':[{'reset_count':5}],'validation':[{'reset_count':1}]}
        runtime.save(folder,task_cursors=cursors)
        fresh=RuntimeV83(self.config());fresh.load(folder)
        self.assertEqual(fresh.value_table.state_dict(),runtime.value_table.state_dict())
        self.collected(runtime,6);self.collected(fresh,6)
        self.assertEqual(runtime.credits,fresh.credits)
        self.assertEqual(runtime.keys,fresh.keys)
        self.assertEqual(runtime.value_snapshot.diagnostics,fresh.value_snapshot.diagnostics)
        # Stored history is protected by the native completion marker digest.
        path=folder/'v83_runtime.json';payload=json.loads(path.read_text())
        payload['tier2_state']['last_committed_iteration']=4;path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):RuntimeV83(self.config()).load(folder)


if __name__=='__main__':unittest.main()
