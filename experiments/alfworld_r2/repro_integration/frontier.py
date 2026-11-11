"""Family-neutral current progress, never an eventual recovery label."""
import json


def frontier_key(*, held, transformed, location_known, at_object, object_open,
                 at_goal, goal_open):
    stage = ("TRANSFORMED_HELD" if transformed else "HELD") if held else (
        "TRANSFORMED_LOCATED" if transformed else "LOCATED") if location_known else "SEARCH"
    return json.dumps({"stage": stage, "required_property": bool(transformed),
                       "location_known": bool(location_known),
                       "at_object": bool(at_object), "object_accessible": bool(object_open),
                       "at_goal": bool(at_goal), "goal_accessible": bool(goal_open)},
                      sort_keys=True, separators=(",", ":"))
