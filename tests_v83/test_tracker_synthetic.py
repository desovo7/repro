"""Semantic state-tracking checks, independent of credit injection code."""
import json
from pathlib import Path
import tempfile
import unittest

from repro_v83.tracker import Task, TrajectoryStateTracker, is_search, prior_potential


class TrackerTests(unittest.TestCase):
    def tracker(self, family="simple", target="pencil", goal="desk"):
        tracker = TrajectoryStateTracker(Task(family, target, goal))
        tracker.reset("You are in the middle of a room. You see a desk 1.")
        return tracker

    def go(self, tracker, location, contents="nothing"):
        return tracker.step("go to " + location,
                            f"You arrive at {location}. On the {location}, you see {contents}.")

    def take(self, tracker, obj, recep):
        return tracker.step(f"take {obj} from {recep}", f"You pick up the {obj} from the {recep}.")

    def put(self, tracker, obj, recep):
        return tracker.step(f"move {obj} to {recep}", f"You move the {obj} to the {recep}.")

    def test_exact_class_matching_and_confirmed_conflict_recovery(self):
        t = self.tracker()
        self.go(t, "desk 1", "a pen 1, and a pencil 1")
        self.assertEqual(t.state.seen_target_locs, {"pencil 1": "desk 1"})
        took = self.take(t, "pen 1", "desk 1")
        self.assertEqual((took.before.p, took.after.p), (0, -1))
        self.assertAlmostEqual(prior_potential(t.task, took.after), -0.15)
        dropped = self.put(t, "pen 1", "desk 1")
        self.assertEqual((dropped.before.p, dropped.after.p), (-1, 0))
        self.assertIsNone(dropped.after.held_obj)
        self.assertEqual(self.take(t, "pencil 1", "desk 1").after.p, 1)

    def test_soapbar_does_not_match_soapbottle(self):
        t = self.tracker(target="soapbar", goal="cart")
        self.go(t, "shelf 1", "a soapbottle 2, and a soapbar 1")
        self.assertEqual(set(t.state.seen_target_locs), {"soapbar 1"})
        self.assertEqual(self.take(t, "soapbottle 2", "shelf 1").after.p, -1)

    def test_failed_or_inconsistent_action_never_changes_hand(self):
        t = self.tracker()
        tr = t.step("take pencil 1 from desk 1", "Nothing happens.")
        self.assertTrue(tr.no_op)
        self.assertIsNone(tr.after.held_obj)
        tr = t.step("take pencil 1 from desk 1", "You pick up the pen 1 from the desk 1.")
        self.assertIsNone(tr.after.held_obj)
        tr = t.step("move pencil 1 to desk 1", "You move the pencil 1 to the desk 1.")
        self.assertFalse(tr.after.delivered_objs)

    def test_processing_is_locked_to_instance_and_correct_tool(self):
        for family, verb, tool in [("clean", "clean", "sinkbasin"),
                                   ("heat", "heat", "microwave"),
                                   ("cool", "cool", "fridge")]:
            with self.subTest(family=family):
                t = self.tracker(family, "mug", "countertop")
                self.go(t, "shelf 1", "a mug 1, and a mug 2")
                self.take(t, "mug 1", "shelf 1")
                bad = t.step(f"{verb} mug 1 with desk 1", f"You {verb} the mug 1 using the desk 1.")
                self.assertEqual(bad.after.p, 1)
                good = t.step(f"{verb} mug 1 with {tool} 1", f"You {verb} the mug 1 using the {tool} 1.")
                self.assertEqual(good.after.p, 2)
                self.put(t, "mug 1", "shelf 1")
                tr = self.take(t, "mug 2", "shelf 1")
                self.assertEqual(tr.after.p, 1)
                self.assertEqual(tr.after.processed_objs, {"mug 1"})
                self.assertEqual(self.put(t, "mug 2", "countertop 1").after.p, -1)

    def test_unprocessed_delivery_conflict_and_retrieval(self):
        t = self.tracker("heat", "mug", "coffeemachine")
        self.take(t, "mug 2", "shelf 1")
        placed = self.put(t, "mug 2", "coffeemachine 1")
        self.assertEqual((placed.before.p, placed.after.p), (1, -1))
        self.assertEqual(placed.after.unprocessed_in_goal, {"mug 2"})
        recovered = self.take(t, "mug 2", "coffeemachine 1")
        self.assertEqual((recovered.before.p, recovered.after.p), (-1, 1))
        t.step("heat mug 2 with microwave 1", "You heat the mug 2 using the microwave 1.")
        complete = self.put(t, "mug 2", "coffeemachine 1")
        self.assertEqual(complete.after.p, 3)
        self.assertEqual(complete.after.delivered_objs, {"mug 2"})

    def test_pick_two_counts_distinct_instances_and_recycling_regresses(self):
        t = self.tracker("pick_two", "pillow", "sofa")
        self.go(t, "armchair 1", "a pillow 1, and a pillow 2")
        self.assertEqual(self.take(t, "pillow 1", "armchair 1").after.p, 1)
        self.assertEqual(self.put(t, "pillow 1", "sofa 1").after.p, 2)
        recycling = self.take(t, "pillow 1", "sofa 1")
        self.assertEqual((recycling.before.p, recycling.after.p), (2, 1))
        self.assertFalse(recycling.after.delivered_objs)
        self.assertEqual(self.put(t, "pillow 1", "sofa 1").after.p, 2)
        self.assertEqual(self.take(t, "pillow 2", "armchair 1").after.p, 3)
        final = self.put(t, "pillow 2", "sofa 1")
        self.assertEqual(final.after.p, 4)
        self.assertEqual(final.after.delivered_objs, {"pillow 1", "pillow 2"})

    def test_navigation_uses_candidate_memory_before_new_observation(self):
        t = self.tracker("pick_two", "pillow", "sofa")
        first = self.go(t, "armchair 1", "a pillow 1, and a pillow 2")
        self.assertEqual(first.after.nav_bit, 1)
        self.assertFalse(first.after.known_candidates)
        self.take(t, "pillow 1", "armchair 1")
        self.go(t, "sofa 1")
        self.put(t, "pillow 1", "sofa 1")
        distracted = self.go(t, "bed 1")
        self.assertEqual(distracted.after.p, 2)
        self.assertEqual(distracted.after.known_candidates, {"armchair 1"})
        self.assertEqual(distracted.after.nav_bit, 0)
        returned = self.go(t, "armchair 1", "a pillow 2")
        self.assertEqual(returned.after.nav_bit, 1)
        self.assertEqual(returned.after.known_candidates, {"armchair 1"})

    def test_open_container_it_contents_establishes_target_candidate(self):
        t = self.tracker()
        self.go(t, "drawer 1")
        opened = t.step("open drawer 1", "You open the drawer 1. The drawer 1 is open. In it, you see a pencil 2.")
        self.assertEqual(opened.after.seen_target_locs, {"pencil 2": "drawer 1"})
        self.go(t, "drawer 2")
        returned = self.go(t, "drawer 1", "a pencil 2")
        self.assertEqual(returned.after.nav_bit, 1)

    def test_complete_listing_invalidates_stale_location(self):
        t = self.tracker()
        self.go(t, "drawer 1", "a pencil 2")
        self.go(t, "shelf 1")
        self.go(t, "drawer 1", "nothing")
        self.assertNotIn("pencil 2", t.state.seen_target_locs)

    def test_ambiguous_in_it_does_not_clear_current_location_memory(self):
        t = self.tracker("heat", "mug", "coffeemachine")
        self.go(t, "countertop 1", "a mug 1, and a mug 2")
        inspected = t.step("examine mug 1", "This is a normal mug 1. In it, you see nothing.")
        self.assertTrue(inspected.no_op)
        self.assertEqual(inspected.after.seen_target_locs,
                         {"mug 1": "countertop 1", "mug 2": "countertop 1"})
        unanchored = t.step("look", "In it, you see nothing.")
        self.assertEqual(unanchored.after.seen_target_locs, inspected.after.seen_target_locs)
        explicit = t.step("go to countertop 1", "You arrive at countertop 1. In it, you see nothing.")
        self.assertFalse(explicit.after.seen_target_locs)

    def test_lamp_first_use_repeat_and_observed_off(self):
        t = self.tracker("look_at", "pencil", None)
        self.go(t, "desk 1", "a desklamp 1")
        first = t.step("use desklamp 1", "You turn on the desklamp 1.")
        self.assertFalse(first.no_op)
        self.assertTrue(first.first_lamp_activation)
        self.assertEqual(first.after.lit_lamp_locs, {"desk 1"})
        self.assertAlmostEqual(prior_potential(t.task, first.after), 0.1)
        again = t.step("use desklamp 1", "You turn on the desklamp 1.")
        self.assertTrue(again.no_op)
        self.assertFalse(again.first_lamp_activation)
        self.assertIn("lamp_already_on", again.no_op_reasons)
        off = t.step("use desklamp 1", "You turn off the desklamp 1.")
        self.assertFalse(off.no_op)
        self.assertFalse(off.after.lit_lamp_locs)

    def test_known_lamp_return_and_terminal_missing_feedback(self):
        t = self.tracker("look_at", "pencil", None)
        self.go(t, "desk 1", "a desklamp 1")
        t.step("use desklamp 1", "You turn on the desklamp 1.")
        self.go(t, "shelf 1", "a pencil 3")
        self.take(t, "pencil 3", "shelf 1")
        self.assertEqual(t.state.p, 1)
        final = t.step("go to desk 1", None, won=True)
        self.assertEqual((final.before.p, final.after.p), (1, 3))
        self.assertTrue(final.feedback_missing)
        self.assertFalse(final.no_op)
        self.assertEqual(final.after.loc, "shelf 1")  # no invented arrival observation
        self.assertGreater(prior_potential(t.task, final.after), prior_potential(t.task, final.before))

    def test_won_progress_priority_does_not_override_noop_predicate(self):
        t = self.tracker()
        self.take(t, "pen 1", "desk 1")
        final = t.step("examine pen 1", "Nothing happens.", won=True, format_valid=False)
        self.assertEqual(final.after.p, 2)
        self.assertTrue(final.no_op)
        self.assertIn("environment_rejection", final.no_op_reasons)
        self.assertIn("invalid_format", final.no_op_reasons)

    def test_failed_final_missing_feedback_does_not_invent_action_effect(self):
        t = self.tracker()
        final = t.step("take pencil 1 from desk 1", None)
        self.assertTrue(final.feedback_missing)
        self.assertIsNone(final.after.held_obj)
        self.assertEqual(final.after.p, 0)
        self.assertEqual(final.after.nav_bit, 0)

    def test_noop_rejection_inspection_text_invariance_and_invalid_format(self):
        t = self.tracker()
        self.go(t, "desk 1")
        for action, observation, valid, expected in [
                ("examine desk 1", "There is nothing special about the desk 1.", True, "redundant_inspection"),
                ("take pencil 1 from desk 1", "Nothing happens.", True, "environment_rejection"),
                ("look", "Nothing happens.", True, "text_invariance"),
                ("invalid", "Some new text.", False, "invalid_format")]:
            with self.subTest(action=action):
                tr = t.step(action, observation, format_valid=valid)
                self.assertTrue(tr.no_op)
                self.assertIn(expected, tr.no_op_reasons)
                self.assertEqual(tr.after.nav_bit, 0)

    def test_task_metadata_portability_and_validation(self):
        task = Task.from_gamefile("/remote/train/pick_heat_then_place_in_recep-Mug-None-CoffeeMachine-10/trial_X/game.tw-pddl")
        self.assertEqual((task.family, task.target_type, task.goal_recep_type, task.tool_recep_type, task.k),
                         ("pick_heat_then_place_in_recep", "mug", "coffeemachine", "microwave", 3))
        task = Task.from_gamefile("look_at_obj_in_light-CD-None-DeskLamp-302/trial_X")
        self.assertEqual((task.target_type, task.goal_recep_type, task.tool_recep_type), ("cd", None, "desklamp"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "traj_data.json"
            path.write_text(json.dumps({"task_type": "pick_and_place_simple", "pddl_params":
                                       {"object_target": "SoapBar", "parent_target": "Cart", "object_sliced": False}}))
            task = Task.from_gamefile(Path(tmp) / "game.tw-pddl")
            self.assertEqual((task.target_type, task.goal_recep_type), ("soapbar", "cart"))
        with self.assertRaises(ValueError):
            Task("unsupported", "pencil", "desk")
        self.assertTrue(is_search(Task("pick_two", "pillow", "sofa"), 2))
        self.assertFalse(is_search(Task("simple", "pillow", "sofa"), 1))

    def test_parser_cache_does_not_leak_state_or_task_identity(self):
        pencil = self.tracker(target="pencil", goal="desk")
        pen = self.tracker(target="pen", goal="desk")
        for tracker in [pencil, pen]:
            self.go(tracker, "shelf 1", "a pen 1, and a pencil 1")
        self.assertEqual(pencil.state.seen_target_locs, {"pencil 1": "shelf 1"})
        self.assertEqual(pen.state.seen_target_locs, {"pen 1": "shelf 1"})
        self.assertEqual(self.take(pencil, "pen 1", "shelf 1").after.p, -1)
        self.assertEqual(self.take(pen, "pen 1", "shelf 1").after.p, 1)
        for _ in range(2):
            lamp = self.tracker("look_at", "pencil", None)
            self.go(lamp, "desk 1", "a desklamp 1")
            first = lamp.step("use desklamp 1", "You turn on the desklamp 1.")
            self.assertTrue(first.first_lamp_activation)
            self.assertFalse(first.no_op)

    def test_reused_immutable_collections_preserve_prior_snapshots(self):
        t = self.tracker("pick_two", "pillow", "sofa")
        self.take(t, "pillow 1", "armchair 1")
        delivered = self.put(t, "pillow 1", "sofa 1").after
        self.take(t, "pillow 1", "sofa 1")
        self.assertEqual(delivered.delivered_objs, {"pillow 1"})
        self.assertFalse(t.state.delivered_objs)
        with self.assertRaises(AttributeError):
            delivered.p = 4


if __name__ == "__main__":
    unittest.main()
