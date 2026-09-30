"""GE slots are only placed where one of their approved courses can be taken (planner.Planner._ge_prereqs).
Synthetic data: GE requirement A is filled by ENG 1, GE requirement B by courses that may need ENG 1.
Run with: python -m unittest discover -s tests -v"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from datastores.common import code_key
from datastores.ge import CAL_GETC, CollegeGE, GECourse, SlotSpec
from datastores.prerequisites import InstitutionPrereqs, PrereqRecord
from planner import AltSpec, Planner, SlotReq
from sep_engine import Course, StudentProfile, Term, generate_sep

F, S, SU = Term.FALL, Term.SPRING, Term.SUMMER


def course_rule(code, timing="before"):
    return {"type": "COURSE", "course_id": f"x:{code_key(code)}", "timing": timing}


def planner(ge_courses: dict[str, str | None], rules: dict[str, dict], completed=(), ap_waived=()) -> Planner:
    """ge_courses: GE course code -> GE-list prerequisite text (None = the list shows none).
    rules: course code -> structured registry rule. Every course has 3 units; MAJ 1..4 is a major chain."""
    major = [Course(code="MAJ 1", title="Major 1", units=4), Course(code="MAJ 2", title="Major 2", units=4, prereqs=["MAJ 1"]),
             Course(code="MAJ 3", title="Major 3", units=4, prereqs=["MAJ 2"]), Course(code="MAJ 4", title="Major 4", units=4)]
    # GE-only courses come from the college catalog (units only), like real data: the agreement's own course list
    # (and its legacy prerequisite lists) covers major prep alone
    college_catalog = {code_key(c): SimpleNamespace(code=c, title=c, units=3.0, type="credit")
                       for c in [*ge_courses, "XYZ 9"]}
    p = Planner(SimpleNamespace(ge_pathway="CAL-GETC", start_term="Fall 2026", include_summer=False), None, None)
    p.pathway = SimpleNamespace(college="Test College", university="Test University", major="Testing", degree="B.S.",
                                catalog={c.code: c for c in major}, file="test.json")
    p.ag, p.catalog, p.college_id, p.ge_pid, p.ge_label = None, college_catalog, "x", CAL_GETC, "Cal-GETC"
    p.completed_real, p.ap_waived = list(completed), list(ap_waived)
    p.prereg = InstitutionPrereqs("x", "test.json", None, records={
        code_key(c): PrereqRecord(code_key(c), c, "documented", r, None, (), True, "registry:test.json")
        for c, r in rules.items()})
    p.ge_ds = CollegeGE("test_ge", "test_ge.json", "x", "Test College", CAL_GETC, "Cal-GETC", "2026-2027", {}, None, [],
                        courses={code_key(c): GECourse(c, c, 3.0, (), (), (), text) for c, text in ge_courses.items()})
    return p


def slot(sid, *codes):
    return SlotReq(SlotSpec(sid, sid, sid, f"Area {sid}", f"Area {sid}", frozenset(code_key(c) for c in codes),
                            None, None, False), None)


def build_and_schedule(p: Planner, variant, include_summer=False, start=(F, 2026)):
    built = p._build(AltSpec(0, ["MAJ 1", "MAJ 2", "MAJ 3", "MAJ 4"], [], [], []), variant, {})
    result = generate_sep(built.catalog, built.agreement, StudentProfile(
        completed_courses=built.completed, start_term=start[0], start_year=start[1], include_summer=include_summer))
    terms = [{c["code"]: s["index"] for s in plan["semesters"] for c in s["courses"]} for plan in result["plans"]]
    return built, terms


PROFILES = [((F, 2026), False), ((S, 2027), False), ((S, 2027), True), ((SU, 2027), True)]


class GESlotPrerequisiteTests(unittest.TestCase):
    def test_slot_waits_for_the_course_its_approved_courses_need(self):
        # GE B's only course (ENG 2) needs ENG 1, which fills GE A: ENG 1 then GE B, never GE B first
        p = planner({"ENG 1": None, "ENG 2": "ENG 1"}, {"ENG 2": course_rule("ENG 1")})
        built, _ = build_and_schedule(p, [slot("A", "ENG 1"), slot("B", "ENG 2")])
        self.assertEqual(built.catalog["GE B"].prereqs, ["ENG 1"])
        self.assertEqual(built.placeholders["GE B"]["planned_after"], ["ENG 1"])
        self.assertIn("ENG 1", next(g for g in built.agreement.requirement_groups if g.id == "ge-A").options)
        for start, summer in PROFILES:
            _, terms = build_and_schedule(p, [slot("A", "ENG 1"), slot("B", "ENG 2")], summer, start)
            for t in terms:
                self.assertLess(t["ENG 1"], t["GE B"], (start, summer, t))
                self.assertNotIn("GE A", t)              # ENG 1 itself fills GE A: no extra course
        self.assertEqual(built.ge_links, [("GE B", "ENG 1")])

    def test_one_free_approved_course_frees_the_slot(self):
        # B1 needs ENG 1, B2 needs nothing: GE B may go anywhere, and B1's prerequisite isn't imposed
        p = planner({"ENG 1": None, "ENG 2": "ENG 1", "PHL 5": None}, {"ENG 2": course_rule("ENG 1")})
        built, _ = build_and_schedule(p, [slot("A", "ENG 1"), slot("B", "ENG 2", "PHL 5")])
        self.assertEqual((built.catalog["GE B"].prereqs, built.catalog["GE B"].coreqs), ([], []))
        self.assertNotIn("ENG 1", built.catalog)
        self.assertEqual(built.ge_links, [])

    def test_registry_none_frees_the_slot(self):
        p = planner({"ENG 1": None, "ENG 2": "ENG 1", "PHL 5": "text only"},
                    {"ENG 2": course_rule("ENG 1"), "PHL 5": {"type": "NONE"}})
        built, _ = build_and_schedule(p, [slot("A", "ENG 1"), slot("B", "ENG 2", "PHL 5")])
        self.assertEqual(built.catalog["GE B"].prereqs, [])

    def test_or_branch_met_by_a_completed_course(self):
        # ENG 2 needs ENG 1 OR XYZ 9; XYZ 9 is done, so nothing is forced (ENG 1 is not added for it)
        rule = {"type": "OR", "items": [course_rule("ENG 1"), course_rule("XYZ 9")]}
        p = planner({"ENG 1": None, "ENG 2": "ENG 1 or XYZ 9"}, {"ENG 2": rule}, completed=["XYZ 9"])
        built, _ = build_and_schedule(p, [slot("A", "ENG 1"), slot("B", "ENG 2")])
        self.assertEqual(built.catalog["GE B"].prereqs, [])
        self.assertNotIn("ENG 1", built.catalog)

    def test_or_uses_the_branch_the_plan_already_has(self):
        # Neither done: ENG 1 fills GE A anyway, so that branch is used and XYZ 9 is never added
        rule = {"type": "OR", "items": [course_rule("XYZ 9"), course_rule("ENG 1")]}
        p = planner({"ENG 1": None, "ENG 2": "XYZ 9 or ENG 1"}, {"ENG 2": rule})
        built, _ = build_and_schedule(p, [slot("A", "ENG 1"), slot("B", "ENG 2")])
        self.assertEqual(built.catalog["GE B"].prereqs, ["ENG 1"])
        self.assertNotIn("XYZ 9", built.catalog)

    def test_and_keeps_every_item(self):
        rule = {"type": "AND", "items": [course_rule("ENG 1"), course_rule("MAJ 1")]}
        p = planner({"ENG 1": None, "ENG 2": "ENG 1 and MAJ 1"}, {"ENG 2": rule})
        built, terms = build_and_schedule(p, [slot("A", "ENG 1"), slot("B", "ENG 2")])
        self.assertEqual(sorted(built.catalog["GE B"].prereqs), ["ENG 1", "MAJ 1"])
        for t in terms:
            self.assertLess(max(t["ENG 1"], t["MAJ 1"]), t["GE B"])

    def test_concurrent_prerequisite_allows_the_same_term(self):
        p = planner({"ENG 1": None, "ENG 2": "ENG 1 (may be concurrent)"},
                    {"ENG 2": course_rule("ENG 1", "before_or_concurrent")})
        built, terms = build_and_schedule(p, [slot("A", "ENG 1"), slot("B", "ENG 2")])
        self.assertEqual((built.catalog["GE B"].prereqs, built.catalog["GE B"].coreqs), ([], ["ENG 1"]))
        for t in terms:
            self.assertLessEqual(t["ENG 1"], t["GE B"])

    def test_completed_or_ap_waived_prerequisite_frees_the_slot(self):
        for kwargs in ({"completed": ["ENG 1"]}, {"ap_waived": ["ENG 1"]}):
            p = planner({"ENG 1": None, "ENG 2": "ENG 1"}, {"ENG 2": course_rule("ENG 1")}, **kwargs)
            built, _ = build_and_schedule(p, [slot("B", "ENG 2")])
            self.assertEqual(built.catalog["GE B"].prereqs, [], kwargs)

    def test_condition_route_is_reviewed_not_forced(self):
        # ENG 2: ENG 1 or an equivalent (a condition). With GE A already met, ENG 1 isn't in the plan: no course is
        # forced in, and the slot carries a review item instead
        rule = {"type": "OR", "items": [course_rule("ENG 1"), {"type": "CONDITION", "description": "An equivalent course"}]}
        p = planner({"ENG 1": None, "ENG 2": "ENG 1 or equivalent"}, {"ENG 2": rule})
        built, _ = build_and_schedule(p, [slot("B", "ENG 2")])
        self.assertEqual(built.catalog["GE B"].prereqs, [])
        self.assertTrue(any(r.course == "GE B" and r.kind == "condition_route" for r in built.resolved.reviews))

    def test_prerequisite_text_without_a_structured_rule_is_not_guessed(self):
        # The GE list names a prerequisite but the registry has no rule: no guessed edge, a review instead
        p = planner({"ENG 1": None, "ENG 2": "ENG 1"}, {})
        built, _ = build_and_schedule(p, [slot("A", "ENG 1"), slot("B", "ENG 2")])
        self.assertEqual(built.catalog["GE B"].prereqs, [])
        self.assertTrue(any(r.course == "GE B" and "not in the structured prerequisite data" in r.message
                            for r in built.resolved.reviews))

    def test_prerequisite_outside_ge_is_added_with_a_note(self):
        # ENG 2 needs XYZ 9, which fills no GE requirement and isn't planned: it is added (rule 4 of the resolver)
        p = planner({"ENG 2": "XYZ 9"}, {"ENG 2": course_rule("XYZ 9"), "XYZ 9": {"type": "NONE"}})
        built, terms = build_and_schedule(p, [slot("B", "ENG 2")])
        self.assertEqual(built.catalog["GE B"].prereqs, ["XYZ 9"])
        self.assertEqual(built.resolved.added.get("XYZ 9"), "GE B")
        for t in terms:
            self.assertLess(t["XYZ 9"], t["GE B"])


if __name__ == "__main__":
    unittest.main()
