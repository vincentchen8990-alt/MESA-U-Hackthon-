"""Completed-courses search (datastores/course_search.py, GET /courses/search) on the real Chabot catalog, and the
path from a picked course to POST /generate-sep. Run with: python -m unittest discover -s tests -v"""

from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient

import main
from datastores import DataRegistry
from datastores.catalog import CatalogStore
from datastores.course_search import parse_query

DATA = DataRegistry().load()
CAT = DATA.catalogs.for_institution("chabot")
BODY = {"college": "Chabot College", "university": "Cal State East Bay", "major": "Computer Science, B.S.",
        "ge_pathway": "CAL-GETC", "completed_courses": [], "start_term": "Fall 2026", "include_summer": False}


def codes(result) -> list[str]:
    return [c["code"] for c in result["courses"]]


def scheduled(body) -> set[str]:
    return {c["code"] for plan in body["plans"] for s in plan["semesters"] for c in s["courses"]}


class SubjectResolutionTests(unittest.TestCase):
    def assertSubject(self, query, code, match, only=True):
        r = CAT.search(query, limit=30)
        self.assertEqual((r["subjects"][0]["code"], r["subjects"][0]["match"]), (code, match), query)
        if only:
            self.assertTrue(r["courses"] and all(c["subject"] == code for c in r["courses"]), query)
        return r

    def test_subject_names_come_from_the_catalog_data(self):
        self.assertEqual(CAT.subject_names["MTH"], "Mathematics")
        self.assertEqual(CAT.subject_names["CSCI"], "Computer Science")
        self.assertEqual(CAT.subject_names["BIOS"], "Life Sciences")

    def test_mth_is_mathematics(self):
        r = self.assertSubject("mth", "MTH", "code")
        self.assertEqual(r["subjects"][0]["name"], "Mathematics")
        self.assertEqual(codes(r)[:3], ["MTH 1", "MTH 2", "MTH 3"])            # course-number order

    def test_math_alias_lists_mathematics_not_titles_elsewhere(self):
        self.assertSubject("math", "MTH", "alias")
        self.assertSubject("Mathematics", "MTH", "name")

    def test_compu_resolves_computer_science_first(self):
        r = CAT.search("compu", limit=30)
        head = r["subjects"][0]
        self.assertEqual((head["code"], head["name"], head["count"], head["match"]),
                         ("CSCI", "Computer Science", 12, "name prefix"))
        self.assertEqual(codes(r)[:3], ["CSCI 6", "CSCI 7", "CSCI 8"])
        self.assertEqual(r["courses"][0]["title"], "Computer Programming for Visual Thinkers")

    def test_computer_science_spellings(self):
        for q in ("computer", "comp sci", "CS", "cs", "computer science"):
            self.assertEqual(CAT.search(q)["subjects"][0]["code"], "CSCI", q)

    def test_csci_lists_every_csci_course(self):
        r = self.assertSubject("CSCI", "CSCI", "code")
        self.assertEqual(r["total"], 12)
        self.assertIn("CSCI 15", codes(r))

    def test_other_aliases(self):
        for q, code in (("bio", "BIOS"), ("biology", "BIOS"), ("chem", "CHEM"), ("phys", "PHYS"),
                        ("econ", "ECN"), ("stats", "STAT"), ("psych", "PSY")):
            self.assertEqual(CAT.search(q)["subjects"][0]["code"], code, q)


class CourseSearchTests(unittest.TestCase):
    def test_parse_query_ignores_case_and_spacing(self):
        self.assertEqual(parse_query("mth1"), ("mth", "1"))
        self.assertEqual(parse_query("MTH 1"), ("mth", "1"))
        self.assertEqual(parse_query("engl c1000"), ("engl", "C1000"))
        self.assertEqual(parse_query("C1000"), ("", "C1000"))
        self.assertEqual(parse_query("computer science"), ("computer science", None))

    def test_exact_code_ranks_first(self):
        for q in ("MTH 1", "mth1", "Mth 1", "math 1"):
            r = CAT.search(q)
            self.assertEqual((r["courses"][0]["code"], r["courses"][0]["matched_by"]), ("MTH 1", "code"), q)

    def test_number_filters_inside_the_subject(self):
        self.assertEqual(codes(CAT.search("math 4"))[:2], ["MTH 4", "MTH 41"])

    def test_calculus_title_search(self):
        r = CAT.search("calculus")
        self.assertEqual(codes(r)[:2], ["MTH 1", "MTH 2"])
        self.assertTrue(all("calculus" in c["title"].lower() for c in r["courses"]))
        self.assertTrue(all(c["matched_by"] == "title" for c in r["courses"]))

    def test_common_course_numbers(self):
        self.assertTrue({"ENGL C1000", "STAT C1000", "POLS C1000"} <= set(codes(CAT.search("C1000"))))
        self.assertEqual(codes(CAT.search("engl c1000"))[0], "ENGL C1000")

    def test_former_code_ranks_first(self):
        r = CAT.search("engl 1")
        self.assertEqual((r["courses"][0]["code"], r["courses"][0]["matched_by"]), ("ENGL C1000", "formerly"))

    def test_results_are_grouped_by_subject(self):
        r = CAT.search("compu", limit=30)
        order = [c["subject"] for c in r["courses"]]
        self.assertEqual(order, sorted(order, key=[s["code"] for s in r["subjects"]].index))
        self.assertEqual({s["code"] for s in r["subjects"]}, set(order))

    def test_noncredit_listings_are_not_offered(self):
        self.assertTrue(all("type" not in c for c in CAT.search("esl", limit=100)["courses"]))

    def test_limit_and_empty_query(self):
        self.assertEqual(len(CAT.search("1", limit=7)["courses"]), 7)
        self.assertEqual(CAT.search("  ")["courses"], [])


class SubjectFilterTests(unittest.TestCase):
    def test_programming_within_csci(self):
        r = CAT.search("programming", subject="CSCI")
        self.assertEqual(r["subject_filter"], {"code": "CSCI", "name": "Computer Science", "count": 12})
        self.assertTrue(r["courses"])
        self.assertTrue(all(c["subject"] == "CSCI" and "programming" in c["title"].lower() for c in r["courses"]))
        self.assertIn("CSCI 15", codes(r))

    def test_number_within_subject(self):
        r = CAT.search("1", subject="csci")
        self.assertEqual(codes(r), ["CSCI 10", "CSCI 14", "CSCI 15", "CSCI 19A"])
        self.assertEqual(codes(CAT.search("15", subject="CSCI"))[0], "CSCI 15")

    def test_empty_query_lists_the_whole_subject(self):
        self.assertEqual(len(CAT.search("", subject="CSCI", limit=100)["courses"]), 12)

    def test_letters_naming_the_subject_are_ignored(self):
        self.assertEqual(codes(CAT.search("math 1", subject="MTH"))[0], "MTH 1")

    def test_unknown_subject(self):
        self.assertIsNone(CAT.search("x", subject="ZZZ"))


class SubjectFileTests(unittest.TestCase):
    CATALOG = {"catalog": "Test College 2025-2026 Catalog",
               "courses": [{"id": "ABC 1", "department": "ABC", "title": "Intro", "type": "credit", "units": 3}]}

    def load(self, *files: dict) -> CatalogStore:
        with tempfile.TemporaryDirectory() as d:
            for i, f in enumerate(files):
                Path(d, f"{i}.json").write_text(json.dumps(f), encoding="utf-8")
            store = CatalogStore()
            store.load_directory(Path(d))
        return store

    def test_companion_file_names_subjects(self):
        store = self.load(self.CATALOG, {"catalog": "Test College 2025-2026 Catalog", "subjects": {"abc": "Alphabet"}})
        cat = store.for_institution("test_college")
        self.assertEqual(cat.subject_names, {"ABC": "Alphabet"})
        self.assertEqual(cat.search("alpha")["subjects"][0]["name"], "Alphabet")
        self.assertEqual(store.report.skipped, [])

    def test_unmatched_companion_is_skipped_and_codes_still_work(self):
        store = self.load(self.CATALOG, {"catalog": "Other 2025-2026 Catalog", "subjects": {"ABC": "Alphabet"}})
        self.assertEqual(len(store.report.skipped), 1)
        r = store.for_institution("test_college").search("abc")
        self.assertEqual((r["subjects"][0]["name"], codes(r)), (None, ["ABC 1"]))


class CourseSearchApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def search(self, **params):
        r = self.client.get("/courses/search", params={"institution": "Chabot College", **params})
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def test_payload(self):
        r = self.search(q="compu")
        self.assertEqual((r["institution_id"], r["institution_name"], r["academic_year"]),
                         ("chabot", "Chabot College", "2025-2026"))
        self.assertEqual(r["subjects"][0], {"code": "CSCI", "name": "Computer Science", "count": 12,
                                            "match": "name prefix"})
        self.assertEqual(r["courses"][0], {"code": "CSCI 6", "title": "Computer Programming for Visual Thinkers",
                                           "units": 3.0, "subject": "CSCI", "matched_by": "subject"})

    def test_subject_filter_and_limit(self):
        r = self.search(q="programming", subject="CSCI")
        self.assertTrue(all(c["subject"] == "CSCI" for c in r["courses"]))
        self.assertEqual(len(self.search(q="1", limit=500)["courses"]), 100)      # capped
        self.assertEqual(self.client.get("/courses/search", params={"institution": "chabot", "q": "1",
                                                                    "subject": "ZZZ"}).status_code, 404)

    def test_picked_course_reaches_the_plan_as_completed(self):
        """Search 'CSCI 15' -> pick the top result -> Generate SEP: counted as completed, never scheduled again."""
        before = self.client.post("/generate-sep", json=BODY).json()
        self.assertIn("CSCI 15", scheduled(before))
        picked = self.search(q="CSCI 15")["courses"][0]["code"]
        self.assertEqual(picked, "CSCI 15")
        r = self.client.post("/generate-sep", json={**BODY, "completed_courses": [picked]})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["completed_courses_normalized"],
                         [{"entered": "CSCI 15", "code": "CSCI 15", "matched_by": "agreement"}])
        self.assertNotIn("CSCI 15", scheduled(body))
        self.assertNotIn("CSCI 19A", scheduled(body))        # its OR alternative isn't needed either
        self.assertIn("CSCI 20", scheduled(body))            # later courses still are

    def test_duplicate_completed_course_counts_once(self):
        body = self.client.post("/generate-sep", json={**BODY, "completed_courses": ["CSCI 15", "csci15"]}).json()
        self.assertEqual({r["code"] for r in body["completed_courses_normalized"]}, {"CSCI 15"})
        self.assertNotIn("CSCI 15", scheduled(body))


if __name__ == "__main__":
    unittest.main()
