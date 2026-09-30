"""CSU Golden Four as the third transfer pathway (planner.py, datastores/ge.py, main.py, index.html / ui.js).
Real data: Chabot College -> Cal State East Bay; UC targets use the synthetic UC agreement of test_real_pipeline.
Run with: python -m unittest discover -s tests -v"""

from __future__ import annotations

import re
import unittest
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

import main
from datastores.common import code_key
from datastores.ge import CSU_GOLDEN_FOUR, SlotSpec
from planner import AltSpec, SlotReq, _elective_units

try:                                                    # python -m unittest discover -s tests
    import test_ge_prerequisites as gp
    import test_real_pipeline as rp
except ImportError:                                     # python -m unittest tests.test_golden_four
    from tests import test_ge_prerequisites as gp
    from tests import test_real_pipeline as rp

ROOT = Path(__file__).resolve().parent.parent
CSU = {**rp.BODY, "transfer_pathway": "csu_golden_four"}
CSU.pop("ge_pathway")
GOLDEN_FOUR = ["1A", "1B", "1C", "2"]


def scheduled(plan) -> dict[str, dict]:
    return rp.scheduled(plan)


def slots(plan, role=None) -> list[dict]:
    return [c for c in scheduled(plan).values() if c["type"] == "ge_slot" and (role is None or c["ge"].get("role") == role)]


class GoldenFourCSUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        r = cls.client.post("/generate-sep", json=CSU)
        assert r.status_code == 200, r.text
        cls.body = r.json()

    def post(self, status=200, **overrides):
        r = self.client.post("/generate-sep", json={**CSU, **overrides})
        self.assertEqual(r.status_code, status, r.text)
        return r.json()

    # 1. CSU target + Golden Four -> valid
    def test_csu_target_plans_golden_four(self):
        pw = self.body["pathway"]
        self.assertEqual((pw["transfer_pathway"], pw["ge_pathway"]), ("csu_golden_four", "CSU Golden Four"))
        for plan in self.body["plans"]:
            ge = plan["ge"]
            self.assertEqual((ge["pathway"], ge["display_name"], ge["course_list_name"]),
                             ("csu_golden_four", "CSU Golden Four", "Cal-GETC"))
            self.assertEqual(ge["dataset_id"], "chabot_cal_getc_2025_2026")      # Chabot's own Cal-GETC list
            self.assertEqual(plan["transfer_requirements"]["golden_four"]["status"], "planned")
            self.assertTrue(plan["transfer_requirements"]["golden_four"]["complete_before_transfer"])
        self.assertEqual(self.body["data_sources"]["golden_four"]["file"], "csu_golden_four.json")

    # 3. CSU target + UC 7-course -> invalid (the new field and the legacy label)
    def test_uc_seven_course_is_invalid_for_a_csu_target(self):
        for body in ({**CSU, "transfer_pathway": "uc_seven_course"},
                     {**rp.BODY, "ge_pathway": "7-Course Pattern"}):
            r = self.client.post("/generate-sep", json=body)
            self.assertEqual(r.status_code, 422)
            self.assertIn("UC transfer admission requirement", r.json()["detail"])

    # 5. CAL-GETC -> valid for a CSU target (UC: GoldenFourUCTests)
    def test_cal_getc_is_valid_for_a_csu_target(self):
        body = self.post(transfer_pathway="cal_getc")
        self.assertEqual((body["pathway"]["transfer_pathway"], body["pathway"]["ge_pathway"]), ("cal_getc", "CAL-GETC"))
        self.assertIsNone(body["plans"][0]["transfer_requirements"])
        self.assertEqual(len(body["plans"][0]["ge"]["requirements"]), 11)

    # 6. one pathway per request (the UI's radio group is checked in GoldenFourUITests)
    def test_exactly_one_transfer_pathway(self):
        self.post(422, transfer_pathway="golden_four")                         # not one of the three ids
        self.post(422, transfer_pathway=["cal_getc", "csu_golden_four"])        # never several
        self.post(422, ge_pathway="CAL-GETC")                                   # legacy label naming another one
        legacy = {**rp.BODY, "ge_pathway": "CSU Golden Four"}
        r = self.client.post("/generate-sep", json=legacy)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["pathway"]["transfer_pathway"], "csu_golden_four")
        default = {k: v for k, v in rp.BODY.items() if k != "ge_pathway"}
        self.assertEqual(self.client.post("/generate-sep", json=default).json()["pathway"]["transfer_pathway"], "cal_getc")

    # 7. major prep fills a Golden Four area and counts toward GE units
    def test_major_prep_overlap(self):
        for plan in self.body["plans"]:
            rows = {r["requirement_id"]: r for r in plan["ge"]["requirements"]}
            self.assertEqual(sorted(rows), GOLDEN_FOUR)
            self.assertEqual((rows["2"]["status"], rows["2"]["satisfied_by"]["ref"]), ("major_prep", "MTH 1"))
            codes = scheduled(plan)
            self.assertEqual((codes["MTH 1"]["type"], codes["MTH 1"]["is_ge"]), ("major_prep", False))
            self.assertEqual([a["label"] for a in codes["MTH 1"]["ge_satisfies"]], ["Golden Four Area 2"])
            self.assertNotIn("GE 2", codes)                                     # no separate math GE
            # PHYS 4A (major prep, on the Cal-GETC 5A list) counts toward the 30 GE units: no GE 5A slot
            self.assertEqual([a["label"] for a in codes["PHYS 4A"]["ge_satisfies"]], ["Cal-GETC Area 5A"])
            self.assertNotIn("GE 5A", codes)
            counted = {r["code"]: r for r in plan["transfer_requirements"]["ge_units"]["courses"] if r["code"]}
            self.assertEqual((counted["MTH 1"]["requirement_id"], counted["MTH 1"]["units"]), ("2", 5))
            self.assertEqual((counted["PHYS 4A"]["requirement_id"], counted["PHYS 4A"]["golden_four"]), ("5A", False))

    # 8. completed courses satisfy the Golden Four
    def test_completed_courses_satisfy_golden_four(self):
        body = self.post(completed_courses=["ENGL 1", "COMM C1000"])          # ENGL 1 = former ENGL C1000
        for plan in body["plans"]:
            rows = {r["requirement_id"]: r for r in plan["ge"]["requirements"]}
            self.assertEqual((rows["1A"]["status"], rows["1A"]["satisfied_by"]["ref"]), ("completed", "ENGL C1000"))
            self.assertEqual((rows["1C"]["status"], rows["1C"]["satisfied_by"]["ref"]), ("completed", "COMM C1000"))
            codes = scheduled(plan)
            self.assertFalse({"GE 1A", "GE 1C", "ENGL C1000"} & set(codes))
            self.assertNotIn("planned_after", codes["GE 1B"]["ge"])            # its prerequisite is behind the student
            gu = plan["transfer_requirements"]["ge_units"]
            self.assertEqual(gu["completed"], 7)                                # 4 + 3 units, already earned
            tu = plan["transfer_requirements"]["transferable_units"]
            self.assertEqual(tu["completed"], 7)                                # both are on the Cal-GETC list

    # 9. AP counts only where the college's AP chart states a GE area, and adds no units
    def test_ap_counts_only_where_the_chart_says(self):
        body = self.post(ap_scores=[{"subject": "AP English Language and Composition", "score": 4},
                                    {"subject": "AP Psychology", "score": 4},
                                    {"subject": "AP Computer Science A", "score": 4}])
        by_exam = {e["exam_id"]: e["ge_effect"] for e in body["ap_evaluations"]}
        self.assertEqual((by_exam["ap:english_language_and_composition"]["status"],
                          by_exam["ap:english_language_and_composition"]["areas"]), ("eligible", ["1A"]))
        self.assertEqual(by_exam["ap:computer_science_a"]["status"], "none_listed")   # no GE area on the chart
        for plan in body["plans"]:
            rows = {r["requirement_id"]: r for r in plan["ge"]["requirements"]}
            self.assertEqual(rows["1A"]["status"], "ap")                        # chart: 1A
            self.assertEqual({rows[a]["status"] for a in ("1B", "1C")}, {"planned"})   # nothing else claimed
            gu = plan["transfer_requirements"]["ge_units"]
            self.assertEqual(sorted(gu["ap_areas"]), ["1A", "4"])               # psychology: area 4, not Golden Four
            self.assertTrue(all(r["units"] == 0 for r in gu["courses"] if r["status"] == "ap"))
            self.assertFalse(plan["transfer_requirements"]["transferable_units"]["ap_units_counted"])
            self.assertNotIn("GE 4#1", scheduled(plan))                         # area 4 isn't planned twice
        self.assertTrue(any("add no units" in n["message"] for n in body["notes"]))
        # AP covers areas but adds no units: a total still below 30 then needs review, it is never "met" or "short"
        heavy = self.post(major="Computer Engineering", ap_scores=[{"subject": "AP Calculus BC", "score": 5},
                                                                   {"subject": "AP English Literature and Composition", "score": 4}])
        gu = heavy["plans"][0]["transfer_requirements"]["ge_units"]
        self.assertEqual((gu["met"], gu["status"]), (False, "needs_review"), gu["total"])
        self.assertTrue(any("needs review with a counselor" in n["message"] for n in heavy["notes"]))
        # AP Calculus AB fills Area 2 (chart: "2") and waives MTH 1
        calc = self.post(ap_scores=[{"subject": "AP Calculus AB", "score": 4}])
        rows = {r["requirement_id"]: r for r in calc["plans"][0]["ge"]["requirements"]}
        self.assertEqual(rows["2"]["status"], "ap")
        self.assertNotIn("MTH 1", scheduled(calc["plans"][0]))
        # below the chart's minimum score: no area
        low = self.post(ap_scores=[{"subject": "AP English Language and Composition", "score": 2}])
        rows = {r["requirement_id"]: r for r in low["plans"][0]["ge"]["requirements"]}
        self.assertEqual(rows["1A"]["status"], "planned")

    # 10. Golden Four prerequisite ordering on the real data, every major / start / summer choice
    def test_golden_four_prerequisite_ordering(self):
        bodies = [self.body] + [self.post(major=m) for m in ("Computer Engineering", "Physics", "Biochemistry")]
        bodies += [self.post(start_term="Spring 2027", include_summer=True),
                   self.post(start_term="Summer 2027", include_summer=True)]
        for body in bodies:
            for plan in body["plans"]:
                at = {c: s["index"] for c, s in scheduled(plan).items()}
                self.assertLess(at["ENGL C1000"], at["GE 1B"], (plan["label"], at))
                self.assertEqual(scheduled(plan)["GE 1B"]["ge"]["planned_after"], ["ENGL C1000"])
                for code, c in scheduled(plan).items():
                    for dep in (c.get("ge") or {}).get("planned_after", []):
                        self.assertLess(at[dep], at[code], (code, dep))
                for e in plan["graph"]["edges"]:                                # every prerequisite still holds
                    src, dst = (next(n["data"]["label"] for n in plan["graph"]["nodes"]
                                     if n["data"]["id"] == e["data"][k]) for k in ("source", "target"))
                    if src in at and dst in at:
                        self.assertTrue(at[src] < at[dst] if e["data"]["relation"] == "prereq" else at[src] <= at[dst])
        note = next(n for n in self.body["notes"] if n.get("course") == "GE 1B")
        self.assertIn("ENGL 4A, ENGL C1001", note["message"])

    def test_golden_four_is_scheduled_early_without_delaying_major_prep(self):
        """A soft preference: the four areas are done in the first half of the plan, and the major's prerequisite
        chain still starts in the first term, as it does with CAL-GETC."""
        for major, chain_start in (("Computer Science", "CSCI 14"), ("Physics", "PHYS 3A"), ("Biochemistry", "BIOS 21A")):
            cal = {p["label"]: scheduled(p)[chain_start]["index"] for p in self.post(major=major, transfer_pathway="cal_getc")["plans"]
                   if chain_start in scheduled(p)}
            for plan in self.post(major=major)["plans"]:
                g4 = plan["transfer_requirements"]["golden_four"]
                order = [s["term"] for s in plan["semesters"]]
                self.assertLessEqual(order.index(g4["completion_term"]) + 1, -(-len(order) // 2), (major, plan["label"], g4))
                if plan["label"] in cal:
                    self.assertLessEqual(scheduled(plan)[chain_start]["index"], cal[plan["label"]], (major, plan["label"]))

    def test_six_semester_ge_spread_preserves_chains_and_unit_totals(self):
        """CompE used to have generic GE counts 1,1,0,1,1,3 despite room under the hard cap."""
        plan = self.post(major="Computer Engineering")["plans"][0]
        sems, courses = plan["semesters"], scheduled(plan)
        self.assertEqual(len(sems), 6)
        counts = [sum(c["type"] in ("ge_slot", "ge_course") for c in s["courses"]) for s in sems]
        self.assertTrue(all(1 <= n <= 2 for n in counts), counts)
        self.assertLessEqual(counts[-1], 2, counts)
        self.assertEqual(sum(counts), 8)  # seven placeholders plus the named Area 1A course
        for s, ge_count in zip(sems, counts):
            self.assertLess(ge_count, len(s["courses"]), s)  # mixed with technical coursework
            self.assertLessEqual(s["units"], min(18, s["max_units"]))
            self.assertLessEqual(sum(c["code"].startswith("MTH ") for c in s["courses"]), 2)

        at = {c: row["index"] for c, row in courses.items()}
        self.assertEqual((at["CSCI 15"] - at["CSCI 14"], at["CSCI 20"] - at["CSCI 15"]), (1, 1))
        self.assertLess(at["ENGL C1000"], at["GE 1B"])
        nodes = {n["data"]["id"]: n["data"]["label"] for n in plan["graph"]["nodes"]}
        for edge in plan["graph"]["edges"]:
            e = edge["data"]
            src, dst = nodes[e["source"]], nodes[e["target"]]
            if src in at and dst in at:
                self.assertTrue(at[src] < at[dst] if e["relation"] == "prereq" else at[src] <= at[dst], e)
        for req in plan["requirements"]:
            self.assertEqual(len(req["courses"]), req["choose"])
            self.assertTrue(all(c["code"] in courses for c in req["courses"]))

        gu, tu = (plan["transfer_requirements"][k] for k in ("ge_units", "transferable_units"))
        self.assertTrue(gu["met"] and tu["met"])
        self.assertEqual((gu["total"], tu["total"]), (31, 81))
        self.assertEqual(gu["total"], sum(r["units"] for r in gu["courses"]))
        self.assertEqual(tu["total"], sum(c["units"] for c in courses.values())
                         - sum(r["units"] for r in tu["unverified"]))
        extra = slots(plan, "csu_ge_units")
        self.assertEqual(len(extra), 5)
        self.assertLess(gu["total"] - min(c["units"] for c in extra), gu["required"])
        self.assertNotIn("GE 2", courses)  # major-prep math still supplies Golden Four Area 2
        self.assertEqual(tu["elective_units"], 0)

    # 12. not the full 34-unit Cal-GETC
    def test_not_full_cal_getc(self):
        cal = self.post(transfer_pathway="cal_getc")["plans"][0]
        for plan in self.body["plans"]:
            self.assertEqual(len(plan["ge"]["requirements"]), 4)
            self.assertFalse(plan["ge"]["ge_certification"])
            self.assertFalse(plan["transfer_requirements"]["cal_getc_certification"])
            self.assertIsNone(plan["ge"]["lab"])                                 # no Area 5 lab requirement
            ids = {c["ge"]["slot_id"] for c in slots(plan)}
            self.assertLess(len(ids), len(slots(cal)))
            self.assertTrue({"3B", "4#2", "5B"}.isdisjoint(ids), ids)          # Cal-GETC areas it doesn't need
            self.assertEqual({c["ge"]["slot_id"] for c in slots(plan, "golden_four")}, {"1B", "1C"})
        self.assertTrue(any("isn't Cal-GETC certification" in n["message"] for n in self.body["notes"]))

    # 13. the CSU 30 GE-unit minimum is tracked (and planned) beyond the four areas
    def test_thirty_ge_units_are_tracked(self):
        for plan in self.body["plans"]:
            gu = plan["transfer_requirements"]["ge_units"]
            self.assertEqual(gu["required"], 30)
            self.assertTrue(gu["met"] and gu["total"] >= 30 and gu["status"] == "met", gu)
            self.assertEqual(gu["total"], sum(r["units"] for r in gu["courses"]))
            self.assertEqual({r["requirement_id"] for r in gu["courses"] if r["golden_four"]}, set(GOLDEN_FOUR))
            extra = slots(plan, "csu_ge_units")
            self.assertTrue(extra)                                              # the four areas alone are < 30 units
            self.assertTrue(all("not part of the Golden Four" in c["ge"]["rule"] for c in extra))
            self.assertTrue(all(c["ge"]["pathway"] == CSU_GOLDEN_FOUR for c in extra))
            # just enough: without the last extra slot the total would fall short
            self.assertLess(gu["total"] - min(c["units"] for c in extra), 30)
        # completed GE in other areas replaces extra slots (ES 1 is listed in 4 and 6: it counts toward one)
        base = {c["ge"]["slot_id"] for c in slots(self.body["plans"][0], "csu_ge_units")}
        done = self.post(completed_courses=["ARTH 1", "ES 1"])
        for plan in done["plans"]:
            gu = plan["transfer_requirements"]["ge_units"]
            counted = {r["code"]: r["slot_id"] for r in gu["courses"] if r["status"] == "completed"}
            self.assertEqual(counted["ARTH 1"], "3A")
            self.assertIn(counted["ES 1"], ("4#1", "6"))
            ids = {c["ge"]["slot_id"] for c in slots(plan, "csu_ge_units")}
            self.assertTrue({"3A", counted["ES 1"]}.isdisjoint(ids), ids)
            self.assertEqual(len(ids), len(base) - 2)                           # 6 units earned: two fewer slots
            self.assertEqual(gu["completed"], 6)
            self.assertGreaterEqual(gu["total"], 30)

    # 14. the 60 transferable-unit minimum is tracked; no course is invented to reach it
    def test_sixty_transferable_units_are_tracked(self):
        for plan in self.body["plans"]:
            tu = plan["transfer_requirements"]["transferable_units"]
            self.assertEqual(tu["required"], 60)
            self.assertTrue(tu["met"] and tu["total"] >= 60, tu)
            electives = [c for c in scheduled(plan).values() if c["type"] == "elective_slot"]
            self.assertEqual(sum(c["units"] for c in electives), tu["elective_units"])
            self.assertTrue(electives and all(c["title"] == "CSU-transferable elective" and c["is_ge"] for c in electives))
            self.assertLess(tu["total"] - tu["elective_units"], 60)             # electives only when needed ...
            self.assertEqual(tu["total"], 60)                                   # ... and exactly enough
            self.assertEqual([u["code"] for u in tu["unverified"]], ["CSCI 14"])   # not articulated, not GE-listed
            self.assertFalse(any(n["data"]["label"].startswith("ELECTIVE") for n in plan["graph"]["nodes"]
                                 if not n["data"]["is_ge"]))
        self.assertTrue(any("CSCI 14 (4 units)" in n["message"] for n in self.body["notes"]))
        # enough major prep: no electives
        ce = self.post(major="Computer Engineering")["plans"][0]
        self.assertEqual(ce["transfer_requirements"]["transferable_units"]["elective_units"], 0)
        self.assertFalse(any(c["type"] == "elective_slot" for c in scheduled(ce).values()))

    def test_unknowns_are_reported_not_assumed(self):
        tr = self.body["plans"][0]["transfer_requirements"]
        self.assertEqual((tr["gpa"]["status"], tr["gpa"]["minimum"], tr["gpa"]["minimum_nonresident"]),
                         ("not_evaluated", 2.0, 2.4))
        self.assertEqual(tr["good_standing"]["status"], "not_evaluated")
        self.assertEqual(tr["campus_requirements"]["status"], "needs_review")
        self.assertEqual(tr["golden_four"]["minimum_grade"], "C-")
        self.assertEqual(tr["major_preparation"]["remaining_after_transfer"],
                         [r["label"] for r in self.body["remaining_after_transfer"]])
        self.assertTrue(tr["major_preparation"]["remaining_after_transfer"])

    def test_ge_options_for_golden_four_slots_come_from_the_cal_getc_list(self):
        ds = main.DATA.ge.college_list("chabot", "cal_getc", "2026-2027")
        for c in slots(self.body["plans"][0]):
            g = c["ge"]
            data = self.client.get("/ge/options", params={"institution": g["institution_id"], "pathway": g["pathway"],
                                                          "requirement_id": g["requirement_id"],
                                                          "start_term": "Fall 2026"}).json()
            self.assertEqual((data["status"], data["pathway"], data["display_name"]),
                             ("loaded", "csu_golden_four", "Cal-GETC"))
            eligible = next(s.eligible for s in ds.slots if s.requirement_id == g["requirement_id"])
            self.assertTrue(data["options"] and all(code_key(o["code"]) in eligible for o in data["options"]))

    def test_tag_sees_the_pattern_without_electives(self):
        body = self.post(tag_university="davis")
        self.assertTrue(body["tag_evaluations"])
        profile = main.tag_profile(main.GenerateSEPRequest(**{**CSU, "tag_university": "davis"}), body["plans"][0])
        self.assertEqual(profile.ge.pattern, "CSU Golden Four")


class GoldenFourUCTests(unittest.TestCase):
    """UC targets (the synthetic UC agreement from test_real_pipeline.UCTargetTests)."""

    setUp = rp.UCTargetTests.setUp

    def post(self, pathway):
        return self.client.post("/generate-sep", json={**{k: v for k, v in self.body.items() if k != "ge_pathway"},
                                                       "transfer_pathway": pathway})

    # 2. UC target + Golden Four -> invalid
    def test_golden_four_is_invalid_for_a_uc_target(self):
        r = self.post("csu_golden_four")
        self.assertEqual(r.status_code, 422)
        self.assertIn("CSU transfer admission minimum", r.json()["detail"])

    # 4. UC target + UC 7-course -> valid
    def test_uc_seven_course_is_valid_for_a_uc_target(self):
        r = self.post("uc_seven_course")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["pathway"]["transfer_pathway"], "uc_seven_course")
        self.assertIsNone(r.json()["plans"][0]["transfer_requirements"])

    # 5. CAL-GETC -> valid for a UC target
    def test_cal_getc_is_valid_for_a_uc_target(self):
        r = self.post("cal_getc")
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["pathway"]["transfer_pathway"], "cal_getc")


class GoldenFourSyntheticTests(unittest.TestCase):
    """11. Ordering comes from each candidate course's own prerequisites, never from '1A before 1B'."""

    def planner(self, ge_courses, rules):
        p = gp.planner(ge_courses, rules)
        p.ge_pid, p.golden_four, p.ge_label = CSU_GOLDEN_FOUR, main.DATA.ge.golden_four, "CSU Golden Four"
        return p

    @staticmethod
    def slot(sid, *codes):
        return SlotReq(SlotSpec(sid, sid, sid[0], f"Area {sid}", f"Area {sid[0]}",
                                frozenset(code_key(c) for c in codes), None, None, False), None, "golden_four")

    def test_1b_candidate_needing_the_1a_course_is_never_first(self):
        p = self.planner({"ENG 1": None, "ENG 2": "ENG 1"}, {"ENG 2": gp.course_rule("ENG 1")})
        variant = [self.slot("1A", "ENG 1"), self.slot("1B", "ENG 2")]
        for start, summer in gp.PROFILES:
            built, terms = gp.build_and_schedule(p, variant, include_summer=summer, start=start)
            self.assertEqual(built.placeholders["GE 1B"]["planned_after"], ["ENG 1"])
            self.assertEqual(built.placeholders["GE 1B"]["role"], "golden_four")
            for t in terms:
                self.assertLess(t["ENG 1"], t["GE 1B"], t)

    def test_no_prerequisite_means_no_order(self):
        """A 1B course with no prerequisite in the data gets no edge: 1B may share the first term with 1A."""
        p = self.planner({"ENG 1": None, "CRT 1": None}, {"CRT 1": {"type": "NONE"}})
        built = p._build(AltSpec(0, ["MAJ 1", "MAJ 2", "MAJ 3", "MAJ 4"], [], [], []),
                         [self.slot("1A", "ENG 1"), self.slot("1B", "CRT 1")], {})
        self.assertNotIn("planned_after", built.placeholders["GE 1B"])
        self.assertEqual(built.catalog["GE 1B"].prereqs, [])

    def test_golden_four_slots_carry_the_early_preference(self):
        p = self.planner({"ENG 1": None}, {})
        built = p._build(AltSpec(0, ["MAJ 1"], [], [], []), [self.slot("1A", "ENG 1")], {})
        self.assertGreater(built.catalog["GE 1A"].early, 0)
        self.assertEqual(built.catalog["MAJ 1"].early, 0)

    def test_elective_units_add_up_exactly(self):
        self.assertEqual(_elective_units(10), [4.0, 3.0, 3.0])
        self.assertEqual(_elective_units(12), [3.0] * 4)
        self.assertEqual(_elective_units(1), [3.0])
        self.assertEqual(sum(_elective_units(14)), 14)


class GoldenFourUITests(unittest.TestCase):
    """6. The form: one radio group of three pathways, sent as transfer_pathway."""

    html = (ROOT / "index.html").read_text(encoding="utf-8")
    ui = (ROOT / "ui.js").read_text(encoding="utf-8")

    def test_three_options_in_one_radio_group(self):
        radios = re.findall(r'<input type="radio" name="(\w+)" value="(\w+)"', self.html)
        group = [v for n, v in radios if n == "transferPathway"]
        self.assertEqual(group, ["cal_getc", "uc_seven_course", "csu_golden_four"])   # same name: one checked at most
        self.assertEqual(len(re.findall(r'name="transferPathway"[^>]*checked', self.html)), 1)
        self.assertIn('<legend class="block text-sm font-medium mb-2">Transfer pathway</legend>', self.html)
        for title, sub in (("CAL-GETC", "UC + CSU · 34 units"), ("UC 7-Course Pattern", "UC admission minimum"),
                           ("CSU Golden Four", "CSU admission minimum")):
            self.assertIn(f">{title}</span>", self.html)
            self.assertIn(f">{sub}</span>", self.html)
        self.assertNotIn('name="gePathway"', self.html)

    def test_request_and_target_revalidation(self):
        self.assertIn("transfer_pathway: state.pathway", self.ui)
        self.assertNotIn("ge_pathway: state.pathway", self.ui)
        fn = self.ui[self.ui.index("function renderPathwayChoice"):self.ui.index("/* Starting semester")]
        self.assertIn("state.pathway = 'cal_getc'", fn)                           # an invalid choice falls back
        self.assertIn("uc_seven_course: { label: 'UC 7-Course Pattern', system: 'UC'", self.ui)
        self.assertIn("csu_golden_four: { label: 'CSU Golden Four', system: 'CSU'", self.ui)
        on_target_change = self.ui[self.ui.index("function onInstitutionChange"):self.ui.index("function updateGenerateEnabled")]
        self.assertIn("renderPathwayChoice();", on_target_change)             # target changes revalidate the choice


if __name__ == "__main__":
    unittest.main()
