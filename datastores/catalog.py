"""
Community-college course catalogs (data/catalogs/*.json): code, title, units, description and the catalog's
prerequisite TEXT.

The catalog's `prerequisite_course_ids` lists every code a prerequisite sentence mentions, AND or OR alike, so
it is display/support text only; scheduling edges come from the prerequisite registry (prerequisites.py).

A catalog file is recognized by content: a `catalog` title like "Chabot College 2025-2026 Catalog" and a
`courses` list. The title gives the institution (resolved through the shared institution registry) and the
academic year. A companion file with the same `catalog` title and a `subjects` map (department code -> name,
e.g. chabot_2025_2026_subjects.json) names the catalog's subjects for course search (course_search.py).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from .common import CLEAN_CODE, DATA_DIR, LoadReport, code_key, read_json
from .course_search import SubjectIndex

log = logging.getLogger("uvicorn.error")
CATALOG_DIR = DATA_DIR / "catalogs"
_TITLE = re.compile(r"^(?P<name>.+?)\s+(?P<year>\d{4}-\d{4})\s+(?:General\s+)?Catalog\b", re.IGNORECASE)


@dataclass(frozen=True)
class CatalogCourse:
    code: str
    title: str
    units: float | None          # min units (None = noncredit / not stated)
    units_max: float | None
    type: str                    # credit | noncredit | apprenticeship
    department: str
    description: str | None
    prerequisite_text: str | None
    corequisite_text: str | None
    advisory_text: str | None
    formerly: tuple[str, ...]
    page: int | None

    def compact(self) -> dict:
        out = {"code": self.code, "title": self.title, "units": self.units}
        if self.units_max is not None and self.units_max != self.units:
            out["units_max"] = self.units_max
        if self.type != "credit":
            out["type"] = self.type
        return out

    def detail(self) -> dict:
        return {**self.compact(), "type": self.type, "description": self.description,
                "prerequisite_text": self.prerequisite_text, "corequisite_text": self.corequisite_text,
                "advisory_text": self.advisory_text, "formerly": list(self.formerly), "catalog_page": self.page}


@dataclass
class InstitutionCatalog:
    institution_id: str
    institution_name: str
    academic_year: str
    title: str
    file: str
    courses: dict[str, CatalogCourse] = field(default_factory=dict)       # code_key -> course (credit wins)
    former: dict[str, str] = field(default_factory=dict)                  # old code key -> current code
    aliases: dict[str, str] = field(default_factory=dict)                 # same-as code key -> code (other stores)
    subject_names: dict[str, str] = field(default_factory=dict)           # department code -> name (MTH -> Mathematics)

    def get(self, code: str) -> CatalogCourse | None:
        return self.courses.get(code_key(code))

    def resolve(self, raw: str) -> tuple[CatalogCourse, str] | None:
        """(course, how it matched) for any spelling: exact code, a former code ('ENGL 1' -> ENGL C1000)."""
        key = code_key(raw)
        if key in self.courses:
            return self.courses[key], "code"
        if key in self.former and code_key(self.former[key]) in self.courses:
            return self.courses[code_key(self.former[key])], "formerly"
        return None

    @cached_property
    def index(self) -> SubjectIndex:
        """Subject index over the credit courses, built on first search (after every loader has run)."""
        return SubjectIndex(self.courses.values(), self.subject_names, self.former)

    def search(self, q: str, limit: int = 20, subject: str | None = None) -> dict | None:
        """Subject-aware search: {total, subject_filter, subjects: [{code, name, count, match}], courses}.
        With `subject`, that subject only (None if the catalog has no such subject)."""
        if subject:
            return self.index.search_in_subject(subject, q, limit)
        return self.index.search(q, limit)


def _units(raw: Any) -> tuple[float | None, float | None]:
    if isinstance(raw, dict) and raw.get("min") is not None:
        return float(raw["min"]), float(raw.get("max", raw["min"]))
    if isinstance(raw, (int, float)):
        return float(raw), float(raw)
    return None, None


def _subject_names(raw: Any) -> dict[str, str]:
    """{'mth': 'Mathematics'} -> {'MTH': 'Mathematics'}; blank names dropped."""
    if not isinstance(raw, dict):
        return {}
    return {str(k).strip().upper(): str(v).strip() for k, v in raw.items() if str(k).strip() and str(v or "").strip()}


class CatalogStore:
    def __init__(self) -> None:
        self.catalogs: dict[str, list[InstitutionCatalog]] = {}      # institution id -> catalogs (by year)
        self.report = LoadReport()

    def load_directory(self, folder: Path = CATALOG_DIR, resolve_institution=None) -> None:
        """resolve_institution(name) -> canonical id (the shared registry); defaults to a slug of the name."""
        self.catalogs, self.report = {}, LoadReport()
        subject_files: list[tuple[Path, dict]] = []
        for path in sorted(Path(folder).glob("*.json")) if Path(folder).is_dir() else []:
            try:
                raw = read_json(path)
            except (OSError, ValueError) as e:
                self.report.skip(path, f"not readable JSON ({e})")
                continue
            if isinstance(raw, dict) and isinstance(raw.get("subjects"), dict) and "courses" not in raw:
                subject_files.append((path, raw))              # attached once every catalog is read
                continue
            m = _TITLE.match(str(raw.get("catalog", ""))) if isinstance(raw, dict) else None
            if not m or not isinstance(raw.get("courses"), list):
                self.report.skip(path, "not a course catalog (needs a 'catalog' title with institution and year, "
                                       "and a 'courses' list)")
                continue
            name = m["name"].strip()
            inst = resolve_institution(name) if resolve_institution else re.sub(r"\W+", "_", name.lower())
            cat = InstitutionCatalog(inst, name, m["year"], raw["catalog"], path.name,
                                     subject_names=_subject_names(raw.get("subjects")))
            for row in raw["courses"]:
                code = str(row.get("id") or "").strip()
                if not code:
                    continue
                lo, hi = _units(row.get("units"))
                formerly = tuple(f for f in (str(row.get("formerly") or "").strip(),) if CLEAN_CODE.match(f.upper()))
                course = CatalogCourse(
                    code=code, title=str(row.get("title") or "").strip(), units=lo, units_max=hi,
                    type=str(row.get("type") or "credit"), department=str(row.get("department") or ""),
                    description=row.get("description"), prerequisite_text=row.get("prerequisite"),
                    corequisite_text=row.get("corequisite"),
                    advisory_text=row.get("strongly_recommended") or row.get("advisory"),
                    formerly=formerly, page=row.get("page"))
                key = code_key(code)
                if key not in cat.courses or (cat.courses[key].type != "credit" and course.type == "credit"):
                    cat.courses[key] = course                     # credit listing wins over a noncredit twin
                for f in formerly:
                    cat.former.setdefault(code_key(f), code)
            self.catalogs.setdefault(inst, []).append(cat)
            self.report.loaded.append({"file": path.name, "institution_id": inst, "institution": name,
                                       "academic_year": m["year"], "courses": len(cat.courses)})
        for path, raw in subject_files:
            title = str(raw.get("catalog", "")).strip().lower()
            cats = [c for cs in self.catalogs.values() for c in cs if c.title.strip().lower() == title]
            if not cats:
                self.report.skip(path, f"subject names for {raw.get('catalog')!r}, but no such catalog is loaded")
            for cat in cats:
                cat.subject_names.update(_subject_names(raw["subjects"]))
        for cats in self.catalogs.values():
            cats.sort(key=lambda c: c.academic_year, reverse=True)
        for s in self.report.skipped:
            log.info("Catalog file %s skipped: %s", s["file"], s["reason"])

    def for_institution(self, inst_id: str, academic_year: str | None = None) -> InstitutionCatalog | None:
        """That year's catalog, else the latest earlier one, else the latest one (catalog text changes slowly;
        the caller reports the year actually used)."""
        cats = self.catalogs.get(inst_id) or []
        if not cats:
            return None
        if academic_year:
            earlier = [c for c in cats if c.academic_year <= academic_year]
            if earlier:
                return earlier[0]
        return cats[0]

    def add_former_codes(self, inst_id: str, pairs: dict[str, str]) -> None:
        """Former codes other datasets list (e.g. the GE list's former_codes); the catalog's own come first."""
        for cat in self.catalogs.get(inst_id, []):
            for old, new in pairs.items():
                if code_key(new) in cat.courses:
                    cat.former.setdefault(code_key(old), cat.courses[code_key(new)].code)

    def status(self) -> list[dict]:
        return [{"institution_id": c.institution_id, "institution": c.institution_name, "academic_year": c.academic_year,
                 "file": c.file, "courses": len(c.courses), "named_subjects": len(c.subject_names), "status": "loaded"}
                for cats in self.catalogs.values() for c in cats]
