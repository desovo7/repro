"""Counterexamples for the WebShop adapter and ALF R2 transfer contracts."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from .adapter import instruction_slots, visible_product, price_relation, option_groups, first_slate
from .core import Config, CreditEngine, parse_trace
from .r2_credit import budget_bucket, empty_stats, add_trace_stats, build_tables, potentials
from .test_webshop import fixture, batch, HOME, RESULTS, ITEM, DETAIL, CLICK, END


def sequence(actions):
    raw = fixture()
    raw['steps'] = raw['steps'][:2]
    before = ITEM
    for action, after, clicks in actions:
        i = len(raw['steps'])
        raw['steps'].append(dict(raw['steps'][0], step_id=i, action_uid=f'change:{i}',
            observation_before=before, observation_after=after, action=action,
            available_before={'clickables': clicks}, environment_done=after == END))
        before = after
    return raw


class AdapterTests(unittest.TestCase):
    def test_range_normalization_title_and_unknown_boundary(self):
        for text, interval in [('Price: $44.99 to $33.98', (33.98,44.99)),
                               ('Price: $1,000.00', (1000.,1000.)), ('Price: unlisted', None)]:
            title, got = visible_product("'Product title' [SEP] '" + text + "'")
            self.assertEqual(title, 'product title')
            self.assertEqual(got, interval)
        self.assertEqual(price_relation((20,30),30), 'within')
        self.assertEqual(price_relation((20,30),25), 'unknown')
        self.assertEqual(price_relation((20,30),19), 'over')

    def test_structured_values_keep_punctuation_and_conflicts_unknown(self):
        roles, limit, skel, conflicts = instruction_slots(
            'Find rug with size: 7 ft 8 in x 10 ft 2 in, and color: blue / gray, and price lower than 50.00 dollars')
        self.assertEqual(roles, {'size':'7 ft 8 in x 10 ft 2 in', 'color':'blue / gray'})
        self.assertEqual(limit, 50)
        roles2, _, skel2, _ = instruction_slots('Find shirt with color: white, and size: large, and price lower than 9 dollars')
        self.assertEqual(skel, skel2)
        roles3, _, _, conflicts = instruction_slots('Find shirt with color: blue, and color: red')
        self.assertIsNone(roles3['color'])
        self.assertEqual(conflicts, {'color'})

    def test_selection_overwrites_repeats_and_detail_memory(self):
        raw = sequence([('click[red]',ITEM,CLICK), ('click[blue]',ITEM,CLICK), ('click[blue]',ITEM,CLICK),
                        ('click[description]',DETAIL,CLICK), ('click[< prev]',ITEM,['< prev','back to search']),
                        ('click[red]',ITEM,CLICK), ('click[buy now]',END,CLICK)])
        t = parse_trace(raw, Config())
        states = [json.loads(s) for s in t['states']]
        self.assertEqual([s[2] for s in states[2:8]], [['available_unselected'],['selected_requested'],
            ['selected_other'],['selected_other'],['selected_other'],['selected_other']])
        self.assertEqual(t['noops'][2:5], [False,False,True])
        self.assertEqual(states[8][2], ['selected_requested'])

    def test_budget_without_comma_or_before_roles_does_not_contaminate_values(self):
        roles, limit, _, _ = instruction_slots('Find shirt with color: red and price lower than 50 dollars')
        self.assertEqual((roles,limit), ({'color':'red'},50.))
        roles, limit, _, _ = instruction_slots('Find shirt price lower than 50 dollars with color: red, and size: large')
        self.assertEqual((roles,limit), ({'color':'red','size':'large'},50.))
        roles, limit, _, _ = instruction_slots('Find shirt with color: red and price lower than 50 dollars, and size: large')
        self.assertEqual((roles,limit), ({'color':'red','size':'large'},50.))

    def test_results_reset_selection_and_first_slate_does_not_expand(self):
        second = RESULTS.replace('b000000001','b000000002')
        raw = sequence([('click[red]',ITEM,CLICK), ('click[< prev]',second,CLICK),
                        ('click[b000000002]',ITEM,['b000000002','back to search']), ('click[buy now]',END,CLICK)])
        t = parse_trace(raw, Config())
        self.assertEqual(json.loads(t['states'][5])[2], ['available_unselected'])
        self.assertEqual(t['trace'].room, ('single b000000001',))
        self.assertEqual(t['trace'].sightings, [('single b000000001', ('color',))])
        self.assertEqual(t['audit']['opens_outside_first_slate'], 1)
        self.assertEqual(t['coverage'][-1], 1.)

    def test_ambiguous_option_click_does_not_claim_target_selected(self):
        item = ITEM.replace("'color'", "'shape' [SEP] 'red' [SEP] 'round' [SEP] 'color'")
        clicks = CLICK + ['round']
        raw = sequence([('click[red]',item,clicks), ('click[buy now]',END,clicks)])
        raw['steps'][1]['observation_after'] = item
        raw['steps'][2]['observation_before'] = item
        t = parse_trace(raw, Config())
        self.assertEqual(json.loads(t['states'][3])[2], ['unresolved'])
        self.assertEqual(t['audit']['ambiguous_accepted_option'], 1)

    def test_capture_current_clickables_is_same_as_legacy_causal_reconstruction(self):
        raw = fixture(long=True)
        legacy = parse_trace(raw, Config())
        for i, step in enumerate(raw['steps'][:-1]):
            step['available_after'] = raw['steps'][i+1]['available_before']
        current = parse_trace(raw, Config())
        for field in ('states','noops','coverage','audit'):
            self.assertEqual(legacy[field], current[field])
        self.assertEqual(legacy['trace'].sightings, current['trace'].sightings)

    def test_first_slate_missing_title_or_price_cannot_borrow_neighbor_price(self):
        text = "'Page 1 (Total results: 3)' [SEP] 'b000000001' [SEP] '$18.0' [SEP] 'b000000002' [SEP] 'Title without price' [SEP] 'b000000003' [SEP] 'Title' [SEP] '$20 to $30'"
        pool = first_slate(text, {'clickables':['b000000001','b000000002','b000000003']})
        self.assertEqual(pool, {'b000000001':'single b000000001','b000000002':'unknown b000000002','b000000003':'range b000000003'})


class MethodTests(unittest.TestCase):
    def test_relative_buckets_preserve_alf_and_declare_webshop(self):
        self.assertEqual([sum(budget_bucket(i,50)==k for i in range(50)) for k in range(3)], [10,15,25])
        self.assertEqual([sum(budget_bucket(i,15)==k for i in range(15)) for k in range(3)], [3,5,7])

    def test_current_mixed_donors_can_support_current_af_without_af_self_donation(self):
        mixed, failed = batch(wins=3), batch()
        for t in failed:
            t['group_uid']='af'
            t['trajectory_uid']='af'+t['trajectory_uid']
            for s in t['steps']:
                s['action_uid']='af'+s['action_uid']
        e=CreditEngine()
        result=e.prepare(mixed+failed,1)
        self.assertEqual(result['diagnostics']['current_mixed_successes'],3)
        self.assertGreater(result['diagnostics']['ranked_states'],0)
        self.assertTrue(all(result['centered'][s['action_uid']]==0 for t in mixed for s in t['steps']))
        # Only mixed-group reveal counts enter this frozen snapshot.
        self.assertTrue(e.table_snapshot['search'])
        self.assertEqual(e.ranks(),({},0,0))
        e.commit(1)
        self.assertEqual(sum(e.history[-1]['reveals'].values()),16)

    def test_offpath_excludes_raw_successful_detours(self):
        t=parse_trace(fixture(success=True,long=True),Config())['trace']
        stats=empty_stats()
        add_trace_stats(stats,t)
        sk=t.skeleton
        stats['occurs'][sk]['never-successful']=10
        for state in t.states:
            stats['occurs'][sk][state]=30
        tables=build_tables(stats,Config(min_success_support=1))
        self.assertEqual(tables.off_path[sk], {'never-successful'})
        t.success=False
        t.states=[t.states[1],'never-successful', 'unseen']
        t.reveals=[[],[],[]]
        phi,_=potentials(t,tables,Config())
        self.assertGreater(phi[0],0)
        self.assertEqual(phi[1:], [0.,0.])


class PinnedRetentionTests(unittest.TestCase):
    def test_diagnostic_points_survive_score_ranking_pruning_and_restart(self):
        import shutil
        from .native_retention import CheckpointRetention, REQUIRED, write_json, digest
        identity=dict(run_id='pinned',config_hash='c',model_contract_hash='m')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            keeper=CheckpointRetention(root,identity,pinned_steps=[2,4])
            for step,score in [(1,.8),(2,.1),(3,.9),(4,.2),(5,.7),(6,.6)]:
                folder=root/f'global_step_{step}'
                for name in REQUIRED:
                    p=folder/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'policy-fixture')
                runtime=dict(identity,iteration_id=step,gradus_state={'last_iteration':step})
                write_json(folder/'native_runtime.json',runtime)
                write_json(folder/'NATIVE_COMPLETE.json',dict(runtime,runtime_sha256=digest(folder/'native_runtime.json'),
                    rng_sha256=digest(folder/'native_driver_rng.pt')))
                keeper.preserve(folder,score)
                shutil.rmtree(folder)
            resumed=CheckpointRetention(root,identity,pinned_steps=[4,2])
            self.assertEqual([r['step'] for r in resumed.state['best']],[3,1,5])
            self.assertEqual({r['step'] for r in resumed._protected(resumed.state)},{1,2,3,4,5,6})
            with self.assertRaises(ValueError):
                CheckpointRetention(root,identity,pinned_steps=[])


if __name__=='__main__':
    unittest.main(verbosity=2)
