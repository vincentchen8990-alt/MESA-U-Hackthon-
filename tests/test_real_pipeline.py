"""The real flow on the real data in data/: Chabot College -> Cal State East Bay, Computer Science B.S., Fall 2026,
CAL-GETC (planner.py through POST /generate-sep). Run with: python -m unittest discover -s tests -v"""

from __future__ import annotations

import re
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

import main
from articulation import PathwayStore

ROOT = Path(__file__).resolve().parent.parent
BODY = {"college": "Chabot College", "university": "Cal State East Bay", "major": "Computer Science, B.S.",
        "ge_pathway": "CAL-GETC", "completed_courses": [], "start_term": "Fall 2026", "include_summer": False}


def scheduled(plan) -> dict[str, dict]:
    return {c["code"]: {**c, "index": s["index"], "term": s["term"]} for s in plan["semesters"] for c in s["courses"]}


class RealFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)
        r = cls.client.post("/generate-sep", json=BODY)
        assert r.status_code == 200, r.text
        cls.body = r.json()

    def post(self, **overrides):
        r = self.client.post("/generate-sep", json={**BODY, **overrides})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    # 1-2. real agreement
    def test_real_agreement(self):
        art = self.body["articulation"]
        self.assertEqual((art["source"], art["file"], art["academic_year"], art["year_status"]),
                         ("assist", "computer_science.json", "2026-2027", "exact"))
        self.assertEqual(art["document_title"], "2026-2027 Computer Science, B.S. Agreement")

    # 3. OR stays OR
    def test_or_articulation_takes_exactly_one_option(self):
        for plan in self.body["plans"]:
            codes = scheduled(plan)
            self.assertEqual(sum(c in codes for c in ("CSCI 15", "CSCI 19A")), 1)
            self.assertEqual(sum(c in codes for c in ("PHYS 4A", "PHYS 7A")), 1)
        rows = {r["id"]: r for r in self.body["articulation"]["requirements"]}
        self.assertEqual(rows["requirement:CS101"]["source_expression"], "(CSCI 15 or CSCI 19A)")
        self.assertEqual(len(rows["requirement:CS101"]["chosen"]), 1)

    # 4. real prerequisites order the plan
    def test_prerequisite_registry_orders_courses(self):
        for plan in self.body["plans"]:
            t = {c: v["index"] for c, v in scheduled(plan).items()}
            self.assertNotIn("MTH 55", t)                       # pre-transfer math: met per AB 705, not scheduled
            self.assertLess(t["CSCI 14"], t["CSCI 15"])
            self.assertLess(t["CSCI 15"], t["CSCI 20"])
            self.assertLess(t["MTH 1"], t["MTH 2"])
            self.assertLessEqual(t["MTH 2"], t["PHYS 4A"])      # MTH 2 may be concurrent (before_or_concurrent)
            self.assertLess(t["MTH 1"], t["PHYS 4A"])
            types = {c: v["type"] for c, v in scheduled(plan).items()}
            self.assertEqual((types["CSCI 14"], types["CSCI 20"]), ("prerequisite", "major_prep"))
        review = [r for r in self.body["review_items"] if r["kind"] == "policy_satisfied"]
        self.assertEqual([r["course"] for r in review], ["CSCI 14"])
        self.assertTrue(any("MTH 55 was not added" in n["message"] for n in self.body["notes"]))
        # MTH 1's placement-dependent rule is never guessed into a course
        self.assertNotIn("MTH 20", scheduled(self.body["plans"][0]))
        self.assertTrue(any(n.get("course") == "MTH 1" for n in self.body["notes"]))

    # 5-6. completed and AP-waived courses disappear
    def test_completed_and_ap_waived_courses_disappear(self):
        body = self.post(completed_courses=["mth1", "CSCI 14", "CSCI 28"],
                         ap_scores=[{"subject": "AP Calculus BC", "score": 5},
                                    {"subject": "AP Physics C: Mechanics", "score": 4}])
        norm = {r["entered"]: r["code"] for r in body["completed_courses_normalized"]}
        self.assertEqual(norm, {"mth1": "MTH 1", "CSCI 14": "CSCI 14", "CSCI 28": "MTH 8"})
        for plan in body["plans"]:
            codes = scheduled(plan)
            for gone in ("MTH 1", "CSCI 14", "MTH 8", "MTH 2", "PHYS 4A", "PHYS 7A", "MTH 55"):
                self.assertNotIn(gone, codes)
            self.assertIn("CSCI 15", codes)
        ap = {e["exam_id"]: e for e in body["ap_evaluations"]}
        self.assertEqual(ap["ap:calculus_bc"]["community_college_effect"]["waived_courses"], ["MTH 2"])
        self.assertEqual(ap["ap:physics_c_mechanics"]["community_college_effect"]["waived_courses"], ["PHYS 4A"])

    def test_ap_calculus_bc_does_not_clear_calculus_i(self):
        body = self.post(ap_scores=[{"subject": "AP Calculus BC", "score": 5}])
        self.assertIn("MTH 1", scheduled(body["plans"][0]))    # the chart lists MTH 2 only
        self.assertNotIn("MTH 2", scheduled(body["plans"][0]))
        self.assertTrue(any("MTH 1 is also cleared" in w for w in body["warnings"]))

    def test_target_campus_ap_credit_is_separate(self):
        body = self.post(ap_scores=[{"subject": "AP Computer Science A", "score": 4}])
        ev = body["ap_evaluations"][0]
        self.assertEqual(ev["target_campus_effect"]["rules"][0]["course_credit"], "CS 101")   # CSUEB's award ...
        self.assertEqual(ev["community_college_effect"]["waived_courses"], ["CSCI 14"])      # ... Chabot's waiver
        self.assertIn("CSCI 15", scheduled(body["plans"][0]))  # CS 101 credit at CSUEB does not complete CSCI 15

    # 7-10. Cal-GETC from Chabot's own list, overlap, GE rows, GE options
    def test_real_cal_getc_with_major_prep_overlap(self):
        for plan in self.body["plans"]:
            ge = plan["ge"]
            self.assertEqual((ge["dataset_id"], ge["academic_year"], ge["year_status"]),
                             ("chabot_cal_getc_2025_2026", "2025-2026", "older_year"))
            rows = {r["slot_id"]: r for r in ge["requirements"]}
            self.assertEqual(len(rows), 11)
            self.assertEqual((rows["2"]["status"], rows["2"]["satisfied_by"]["ref"]), ("major_prep", "MTH 1"))
            self.assertEqual((rows["5A"]["status"], rows["5A"]["satisfied_by"]["ref"]), ("major_prep", "PHYS 4A"))
            self.assertEqual(ge["lab"]["status"], "major_prep")
            slots = [c for c in scheduled(plan).values() if c["type"] == "ge_slot"]
            ids = sorted(c["ge"]["slot_id"] for c in slots)
            self.assertEqual(ids, ["1B", "1C", "3A", "3B", "4#1", "4#2", "5B", "6"])   # no 2 / 5A duplicates
            # 1A is filled by ENGL C1000 by name: both courses that fill 1B need it (see the GE prerequisite test)
            self.assertEqual((rows["1A"]["status"], rows["1A"]["satisfied_by"]["ref"]), ("planned", "ENGL C1000"))
            self.assertTrue(all(c["is_ge"] and c["ge"]["institution_id"] == "chabot" for c in slots))
            self.assertTrue(all(c["requirement_id"] == c["ge"]["requirement_id"] and not c["ge_satisfies"]
                                for c in slots))

    def test_major_prep_ge_overlap_is_structured(self):
        """MTH 1 stays a major-prep course and carries its GE area as metadata (from the GE list, not units)."""
        for plan in self.body["plans"]:
            codes = scheduled(plan)
            mth1, phys = codes["MTH 1"], codes["PHYS 4A"]
            self.assertEqual((mth1["type"], mth1["is_ge"]), ("major_prep", False))
            self.assertEqual([(a["area"], a["label"]) for a in mth1["ge_satisfies"]], [("2", "Cal-GETC Area 2")])
            self.assertEqual([a["area"] for a in phys["ge_satisfies"]], ["5A", "5C"])
            self.assertEqual(codes["CSCI 20"]["ge_satisfies"], [])     # 4 units, but not on the GE list

    def test_ge_is_spread_across_semesters(self):
        """GE fills the room around the prerequisite chain instead of piling into the last terms."""
        for plan in self.body["plans"]:
            counts = [sum(c["type"] in ("ge_slot", "ge_course") for c in s["courses"]) for s in plan["semesters"]]
            self.assertLessEqual(max(counts) - min(counts), 2, counts)
            self.assertLessEqual(max(counts), 3, counts)
            if len(counts) > 4:                                    # enough terms for <= 2 each
                self.assertLessEqual(max(counts), 2, counts)
            self.assertEqual(sum(counts), 9)                       # same GE requirements, just redistributed

    def test_course_series_continue_without_gaps(self):
        """CSCI 15 needs CSCI 14 and CSCI 20 needs CSCI 15: each follows in the very next semester."""
        for plan in self.body["plans"]:
            at = {c: s["index"] for c, s in scheduled(plan).items()}
            self.assertEqual((at["CSCI 15"] - at["CSCI 14"], at["CSCI 20"] - at["CSCI 15"]), (1, 1), plan["label"])

    def test_parallel_math_is_spread_and_series_continue(self):
        """Computer Engineering: MTH 3/4/6 (all after MTH 2) don't pile into one semester, no subject has 3+
        major courses in a term, CSCI 15 follows CSCI 14 at once and CSCI 20 (which unlocks nothing) follows
        within a semester, and GE is spread over the whole plan instead of dumped at the end."""
        plan = self.post(major="Computer Engineering, B.S.")["plans"][0]
        at = {c: s["index"] for c, s in scheduled(plan).items()}
        self.assertGreater(len({at["MTH 3"], at["MTH 4"], at["MTH 6"]}), 1)
        for s in plan["semesters"]:
            subjects = [c["code"].split()[0] for c in s["courses"] if not c["is_ge"]]
            self.assertLessEqual(max((subjects.count(x) for x in subjects), default=0), 2, s)
            self.assertLessEqual(s["units"], plan["max_units_regular"])
        self.assertEqual(at["CSCI 15"] - at["CSCI 14"], 1)
        self.assertIn(at["CSCI 20"] - at["CSCI 15"], (1, 2))
        # GE in every term and never piled up at the end (was 2,0,0,2,3,3, then 1,1,1,2,2,3); ENGL C1000 counts
        ge = [sum(c["type"] in ("ge_slot", "ge_course") for c in s["courses"]) for s in plan["semesters"]]
        self.assertEqual((min(ge), max(ge)), (1, 2), ge)

    def test_ce_series_continue_and_ge_is_not_saved_for_the_end(self):
        """The reported Computer Engineering plan: CSCI 14 -> 15 -> 20 in consecutive terms (CSCI 21, which needs only
        CSCI 14, keeps the series moving), MTH never sits a term out, every term mixes technical courses with GE, and
        the last term isn't a block of GE."""
        plan = self.post(major="Computer Engineering, B.S.")["plans"][0]
        at = {c: s["index"] for c, s in scheduled(plan).items()}
        self.assertEqual((at["CSCI 15"] - at["CSCI 14"], at["CSCI 20"] - at["CSCI 15"]), (1, 1))
        cs_terms = {at[c] for c in ("CSCI 14", "CSCI 15", "CSCI 20", "CSCI 21")}
        self.assertEqual(cs_terms, set(range(min(cs_terms), max(cs_terms) + 1)))       # no term without CSCI inside
        mth_terms = {i for c, i in at.items() if c.startswith("MTH ")}
        self.assertEqual(mth_terms, set(range(min(mth_terms), max(mth_terms) + 1)))
        for s in plan["semesters"]:
            kinds = [c["type"] for c in s["courses"]]
            ge = sum(k in ("ge_slot", "ge_course") for k in kinds)
            self.assertTrue(1 <= ge <= 2 and ge < len(kinds), (s["term"], kinds))     # GE + technical, every term
        for e in plan["graph"]["edges"]:                                                 # still prerequisite-valid
            src, dst = (next(n["data"]["label"] for n in plan["graph"]["nodes"] if n["data"]["id"] == e["data"][k])
                        for k in ("source", "target"))
            if src in at and dst in at:
                self.assertTrue(at[src] < at[dst] if e["data"]["relation"] == "prereq" else at[src] <= at[dst])
        slots = {c["ge"]["slot_id"] for c in scheduled(plan).values() if c["type"] == "ge_slot"}
        self.assertNotIn("2", slots)                                      # MTH 1 already covers Area 2

    def test_ge_slots_come_after_their_courses_prerequisites(self):
        """Cal-GETC 1B (ENGL 4A / ENGL C1001) both need ENGL C1000: in every plan, for every real major, start term
        and summer choice, ENGL C1000 (which fills 1A) comes before GE 1B, never after it or in the same term. The
        generic check: each GE slot is after every course it is planned after."""
        bodies = [self.body] + [self.post(major=m) for m in ("Computer Engineering", "Physics", "Biochemistry")]
        bodies += [self.post(start_term="Spring 2027", include_summer=True), self.post(start_term="Summer 2027",
                                                                                        include_summer=True)]
        for body in bodies:
            for plan in body["plans"]:
                at = {c: s["index"] for c, s in scheduled(plan).items()}
                self.assertLess(at["ENGL C1000"], at["GE 1B"], (plan["label"], at))
                self.assertNotIn("GE 1A", at)                             # ENGL C1000 itself fills 1A
                for code, c in scheduled(plan).items():
                    for dep in (c.get("ge") or {}).get("planned_after", []):
                        self.assertLess(at[dep], at[code], (code, dep))
        note = next(n for n in self.body["notes"] if n.get("course") == "GE 1B")
        self.assertIn("ENGL C1000", note["message"])
        self.assertIn("ENGL 4A, ENGL C1001", note["message"])

    def test_ge_prerequisite_already_met_frees_the_slot(self):
        """ENGL C1000 completed, or waived by AP English Language (Chabot's AP chart): GE 1B has no constraint."""
        for body in (self.post(completed_courses=["ENGL C1000"]), self.post(completed_courses=["ENGL 1"]),
                     self.post(ap_scores=[{"subject": "AP English Language and Composition", "score": 4}])):
            for plan in body["plans"]:
                slot = scheduled(plan)["GE 1B"]
                self.assertNotIn("planned_after", slot["ge"])
                self.assertNotIn("ENGL C1000", scheduled(plan))

    def test_ge_row_options_come_from_the_real_list(self):
        from datastores.common import code_key
        ds = main.DATA.ge.college_list("chabot", "cal_getc", "2026-2027")
        for c in scheduled(self.body["plans"][0]).values():
            if c["type"] != "ge_slot":
                continue
            g = c["ge"]
            r = self.client.get("/ge/options", params={"institution": g["institution_id"], "pathway": g["pathway"],
                                                       "requirement_id": g["requirement_id"], "start_term": "Fall 2026",
                                                       "lab": str(g["lab_required"]).lower()})
            data = r.json()
            self.assertEqual((r.status_code, data["status"], data["institution_id"]), (200, "loaded", "chabot"))
            eligible = next(s.eligible for s in ds.slots if s.requirement_id == g["requirement_id"])
            self.assertTrue(data["options"])
            self.assertTrue(all(code_key(o["code"]) in eligible for o in data["options"]), g["requirement_id"])

    def test_completed_ge_and_ap_ge_credit(self):
        body = self.post(completed_courses=["ENGL 1", "HIS 1"], ap_scores=[{"subject": "AP Psychology", "score": 4}])
        rows = {r["slot_id"]: r for r in body["plans"][0]["ge"]["requirements"]}
        self.assertEqual(rows["1A"]["satisfied_by"]["ref"], "ENGL C1000")          # former code ENGL 1
        self.assertEqual({rows["3B"]["status"], rows["4#1"]["status"]}, {"completed", "ap"})
        slot_ids = [c["ge"]["slot_id"] for c in scheduled(body["plans"][0]).values() if c["type"] == "ge_slot"]
        self.assertNotIn("1A", slot_ids)
        self.assertEqual(len(slot_ids), 6)

    # 11. no-course-articulated requirements are never scheduled
    def test_remaining_after_transfer(self):
        rem = self.body["remaining_after_transfer"]
        self.assertEqual([r["requirement_id"] for r in rem], ["requirement:CS230", "requirement:MATH225"])
        self.assertTrue(all(r["type"] == "remaining_after_transfer" for r in rem))
        for plan in self.body["plans"]:
            for code in scheduled(plan):
                self.assertFalse(re.match(r"^(CS 230|MATH 225)$", code))

    # 12. TAG stays independent of the target university
    def test_tag_is_independent(self):
        body = self.post(tag_university="davis")
        self.assertEqual(body["pathway"]["university"], "California State University, East Bay")
        self.assertTrue(all(t["evaluation"]["campus"]["id"] == "davis" for t in body["tag_evaluations"]))
        self.assertIsNone(self.body["tag_evaluations"])

    # 13. caveats are one collapsible list of plan notes, never stacked warning banners
    def test_notes_are_structured_and_rendered_collapsed(self):
        kinds = {n["kind"] for n in self.body["notes"]}
        self.assertTrue({"prerequisite", "data_year", "remaining_after_transfer"} <= kinds)
        self.assertEqual(self.body["warnings"], [n["message"] for n in self.body["notes"]])
        ui = (ROOT / "ui.js").read_text(encoding="utf-8")
        self.assertIn("planNotesHTML(resp.warnings)", ui)
        notes_fn = ui[ui.index("function planNotesHTML"):ui.index("function dataSourcesHTML")]
        self.assertIn("<details", notes_fn)
        self.assertNotIn("amber", notes_fn)

    # Rules around it
    def test_seven_course_pattern_only_for_uc_targets(self):
        r = self.client.post("/generate-sep", json={**BODY, "ge_pathway": "7-Course Pattern"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("UC transfer admission requirement", r.json()["detail"])
        self.assertIsNone(self.body["admission_requirements"])

    def test_older_agreement_year_is_reported(self):
        body = self.post(major="Biochemistry")
        art = body["articulation"]
        self.assertEqual((art["academic_year"], art["requested_academic_year"], art["year_status"]),
                         ("2025-2026", "2026-2027", "older_year"))
        self.assertTrue(any("2025-2026 agreement is used" in w for w in body["warnings"]))
        self.assertEqual(body["data_sources"]["catalog"]["year_status"], "older_year")

    def test_all_real_majors_plan(self):
        for major in ("Computer Engineering", "Physics", "Biochemistry"):
            body = self.post(major=major)
            for plan in body["plans"]:
                self.assertTrue(plan["semesters"])
                self.assertTrue(all(c["type"] in ("major_prep", "prerequisite", "ge_slot", "ge_course")
                                    for s in plan["semesters"] for c in s["courses"]))
        physics = self.post(major="Physics")
        self.assertEqual(physics["articulation"]["alternatives_total"], 165)
        rows = {r["id"]: r for r in physics["articulation"]["requirements"]}
        electives = [r for r in rows.values() if r["status"] == "planned"
                     and any(t["code"] in ("CHEM 100", "CS 100", "CS 101", "CHEM 111", "PHYS 115")
                             for t in r["target_courses"])]
        self.assertGreaterEqual(len(electives), 2)                       # MINIMUM: at least 2 entries / 6 units

    def test_summer_start(self):
        body = self.post(start_term="Summer 2027", include_summer=True)
        for s in body["plans"][0]["semesters"]:
            if s["season"] == "Summer":
                self.assertTrue(all(c["is_ge"] for c in s["courses"]))  # major prep stays out of Summer

    def test_unknown_completed_course_is_reported(self):
        body = self.post(completed_courses=["XYZ 99"])
        self.assertTrue(any("XYZ 99" in n["message"] for n in body["notes"] if n["kind"] == "completed_courses"))

    def test_other_college_without_articulation_is_not_supported(self):
        r = self.client.post("/generate-sep", json={**BODY, "college": "Porterville College"})
        self.assertEqual(r.status_code, 404)


class EndpointTests(unittest.TestCase):
    client = TestClient(main.app)

    def test_data_status(self):
        st = self.client.get("/data-status").json()
        self.assertEqual(st["articulation"]["datasets_loaded"], 4)
        self.assertEqual(st["catalogs"]["datasets"][0]["courses"], 1341)
        self.assertEqual(st["prerequisites"]["datasets"][0]["institution_id"], "chabot")
        self.assertEqual(sorted(d["name"] for d in st["ge"]["datasets"]),
                         ["CSU Golden Four", "Chabot College Cal-GETC", "Porterville College Cal-GETC",
                          "UC 7-course pattern"])
        self.assertEqual(len(st["ap"]["datasets"]), 2)
        self.assertEqual(len(st["reference_only"]), 2)
        self.assertEqual(st["tag"]["status"], "loaded")

    def test_institutions_and_majors_from_data(self):
        inst = self.client.get("/institutions").json()
        self.assertEqual([c["id"] for c in inst["colleges"]], ["chabot", "porterville"])
        csueb = next(u for u in inst["universities"] if u["id"] == "csueb")
        self.assertEqual((csueb["system"], csueb["has_articulation"]), ("CSU", True))
        majors = self.client.get("/majors", params={"college": "Chabot", "university": "CSUEB"}).json()["majors"]
        self.assertEqual([m["label"] for m in majors], ["Biochemistry, B.A.", "Computer Engineering, B.S.",
                                                        "Computer Science, B.S.", "Physics, B.A."])
        none = self.client.get("/majors", params={"college": "Chabot", "university": "UC Davis"}).json()
        self.assertEqual(none["majors"], [])

    def test_course_search_and_resolve(self):
        r = self.client.get("/courses/search", params={"institution": "chabot", "q": "calc"}).json()
        self.assertIn("MTH 1", [c["code"] for c in r["courses"]])
        self.assertEqual(self.client.get("/courses/resolve", params={"institution": "Chabot College",
                                                                     "code": "engl 1"}).json()["code"], "ENGL C1000")
        self.assertEqual(self.client.get("/courses/resolve", params={"institution": "chabot", "code": "ZZZ 1"}).status_code, 404)
        self.assertEqual(self.client.get("/courses/search", params={"institution": "laney", "q": "x"}).status_code, 404)
        detail = self.client.get("/courses/detail", params={"institution": "chabot", "code": "CSCI 14"}).json()
        self.assertEqual((detail["prerequisite_text"], detail["prerequisite_record"]["status"]), ("MTH 55", "documented"))

    def test_ap_endpoints(self):
        self.assertEqual(len(self.client.get("/ap/exams").json()["exams"]), 44)
        r = self.client.post("/ap/evaluate", json={"college": "Chabot College", "university": "Cal State East Bay",
                                                   "ap_scores": [{"subject": "AP Biology", "score": 4}]}).json()
        ev = r["evaluations"][0]
        self.assertEqual((ev["community_college_effect"]["waived_courses"], ev["ge_effect"]["areas_text"]),
                         (["BIOS 41"], "5B and 5C"))
        self.assertIn(ev["target_campus_effect"]["status"], ("requires_context_and_policy_review", "manual_review"))

    def test_ge_options_for_other_pathways(self):
        uc = self.client.get("/ge/options", params={"institution": "chabot", "requirement_id": "E",
                                                    "pathway": "7-Course Pattern"}).json()
        self.assertEqual(uc["status"], "not_loaded")
        lab = self.client.get("/ge/options", params={"institution": "chabot", "requirement_id": "5B", "lab": "true",
                                                     "academic_year": "2026-2027"}).json()
        self.assertTrue(lab["options"] and all(o["includes_lab"] for o in lab["options"]))


class UCTargetTests(unittest.TestCase):
    """UC targets (no UC articulation is loaded, so a synthetic legacy agreement stands in)."""

    def setUp(self):
        try:
            from test_articulation import dataset           # python -m unittest discover -s tests
        except ImportError:
            from tests.test_articulation import dataset     # python -m unittest tests.test_real_pipeline
        ds = dataset(dataset_id="uc_fixture", agreement={"to_institution_id": "berkeley"},
                     institutions=[{"id": "chabot", "name": "Chabot College"},
                                   {"id": "berkeley", "name": "University of California, Berkeley"}])
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        import json
        Path(tmp.name, "uc.json").write_text(json.dumps(ds), encoding="utf-8")
        store = PathwayStore()
        store.load_directory(tmp.name)
        patcher = mock.patch.object(main, "STORE", store)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)
        self.body = {**BODY, "university": "UC Berkeley", "major": "Computer Science"}

    def test_uc_seven_course_admission_requirement_for_uc_only(self):
        r = self.client.post("/generate-sep", json={**self.body, "ap_scores": [{"subject": "AP Calculus AB", "score": 4}]})
        self.assertEqual(r.status_code, 200, r.text)
        uc = r.json()["admission_requirements"]["uc_seven_course"]
        rows = {x["id"]: x for x in uc["requirements"]}
        self.assertEqual((rows["M"]["status"], rows["E"]["remaining_courses"]), ("met_by_ap", 2))
        self.assertEqual(uc["course_list"], "not_loaded")

    def test_seven_course_pattern_slots_for_uc(self):
        r = self.client.post("/generate-sep", json={**self.body, "ge_pathway": "7-Course Pattern"})
        self.assertEqual(r.status_code, 200, r.text)
        slots = [c for s in r.json()["plans"][0]["semesters"] for c in s["courses"] if c["type"] == "ge_slot"]
        self.assertEqual(sorted(c["ge"]["requirement_id"] for c in slots), ["BR"] * 4 + ["E", "E", "M"])
        self.assertTrue(all(c["ge"]["pathway"] == "uc_seven_course" for c in slots))
        self.assertTrue(any("no college course list" in w for w in r.json()["warnings"]))


if __name__ == "__main__":
    unittest.main()
