"""
Subject-aware course search for the Completed-courses autocomplete (GET /courses/search).

One SubjectIndex per college catalog, built once from its credit courses and its subject names
(data/catalogs/*_subjects.json: department code -> name, from the catalog's table of contents).

TWO-LEVEL SEARCH
  Level 1 - resolve the SUBJECT the letters name. Subject match tiers:
              1 exact subject code     "mth", "MTH"        -> MTH
              2 exact subject name     "mathematics"       -> MTH
              3 subject alias          "math", "comp sci"  -> MTH, CSCI (SUBJECT_ALIASES below)
              4 subject code prefix    "cs"                -> CSCI
              5 subject name prefix    "compu"             -> Computer Science, Computer Application Systems
              6 subject name word      "science"           -> Computer Science, Life Sciences ...
  Level 2 - list COURSES, best first:
              a. courses of the resolved subjects, best subject first, in course-number order;
                 a typed number filters inside the subject ("math 4" -> MTH 4 first, then MTH 40 ...)
              b. exact course code or former code  "engl c1000", "engl 1" -> ENGL C1000 (ranked first)
              c. course-code prefix / number       "C1000"                -> COMM C1000, ENGL C1000 ...
              d. course title (lowest)             "calculus"             -> MTH 1 Calculus I ...
            c-d from other subjects are skipped when the subject is clear (a code, name or alias hit), so
            "math" lists MTH courses, not every title containing "math". Results are grouped by subject.

SUBJECT FILTER (the UI's "Only this subject" chip): search_in_subject() searches one subject only.
  ""  -> all its courses   "1" -> CSCI 1x ...   "programming" -> its courses with a title word "programming..."

Matching ignores case and spacing ("mth1" == "MTH 1"). Official course codes are never renamed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from .common import code_key

# Common names students type for a subject, keyed by the subject's catalog NAME (lowercase), so they apply to
# whichever department code a college uses for it. Codes, names and name prefixes need no entry here.
SUBJECT_ALIASES: dict[str, tuple[str, ...]] = {
    "mathematics": ("math", "maths"),
    "statistics": ("stats",),
    "computer science": ("cs", "comp sci", "computer"),
    "computer information systems": ("cis",),
    "life sciences": ("biology", "bio", "life science"),
    "biology": ("bio",),
    "chemistry": ("chem",),
    "physics": ("phys",),
    "english": ("eng", "writing"),
    "english as a second language": ("esl",),
    "economics": ("econ",),
    "psychology": ("psych",),
    "political science": ("poli sci", "government"),
    "history": ("hist",),
    "sociology": ("soc",),
    "anthropology": ("anthro",),
    "philosophy": ("phil",),
    "geography": ("geog",),
    "geological sciences": ("geology",),
    "communication studies": ("speech", "communications"),
    "theater arts": ("theatre", "drama"),
    "physical education": ("pe",),
    "administration of justice": ("criminal justice",),
}

SUBJECT_MATCH = {1: "code", 2: "name", 3: "alias", 4: "code prefix", 5: "name prefix", 6: "name word"}
CLEAR_TIER = 3              # tiers 1-3 name the subject outright
MULTI_SUBJECT_MIN = 4       # with several subjects, each still lists at least this many courses


def _flat(text: str) -> str:
    """'Comp Sci' -> 'compsci'"""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _prefix_match(query_words: list[str], words: Iterable[str]) -> bool:
    """Every typed word starts some word of the text."""
    words = tuple(words)
    return bool(query_words) and all(any(w.startswith(q) for w in words) for q in query_words)


def parse_query(query: str) -> tuple[str, str | None]:
    """Subject letters and an optional course number:
    'math 1' -> ('math', '1')   'mth1' -> ('mth', '1')   'engl c1000' -> ('engl', 'C1000')
    'computer science' -> ('computer science', None)   '1' -> ('', '1')   'c1000' -> ('', 'C1000')"""
    words = _words(query)
    if not words:
        return "", None
    last = words[-1]
    if re.fullmatch(r"[a-z]?\d[a-z0-9]*", last):
        return " ".join(words[:-1]), last.upper()
    m = re.fullmatch(r"([a-z]+)(\d[a-z0-9]*)", last)
    if m:
        return " ".join([*words[:-1], m[1]]), m[2].upper()
    return " ".join(words), None


@dataclass(frozen=True)
class Subject:
    code: str                       # MTH
    name: str | None                # Mathematics (None when the catalog doesn't name it)
    count: int                      # searchable courses
    aliases: frozenset[str]         # flat spellings: 'math', 'compsci'

    @property
    def label(self) -> str:
        return self.name or self.code

    def match_tier(self, letters: str) -> int | None:
        """How well typed letters ('math', 'comp sci') name this subject: 1 (best) .. 6, or None."""
        words = _words(letters)
        flat = "".join(words)
        if not flat:
            return None
        name_words = _words(self.name or "")
        if flat.upper() == self.code:
            return 1
        if name_words and flat == "".join(name_words):
            return 2
        if flat in self.aliases:
            return 3
        if self.code.startswith(flat.upper()):
            return 4
        if name_words and len(words) <= len(name_words) and all(n.startswith(w) for w, n in zip(words, name_words)):
            return 5
        if name_words and _prefix_match(words, name_words):
            return 6
        return None

    def summary(self) -> dict:
        return {"code": self.code, "name": self.name, "count": self.count}


@dataclass(frozen=True)
class _Entry:
    course: object                  # catalog.CatalogCourse (compact(), code, department, title)
    key: str                        # MTH1
    department: str
    number: str                     # 1, 31S, C1000
    title_words: tuple[str, ...]

    @property
    def order(self) -> tuple:
        """Course-number order: MTH 1, MTH 2 ... MTH 44, MTH 104; C1000 sorts as 1000; no digits last."""
        m = re.search(r"\d+", self.number)
        return (int(m[0]) if m else 10**6, self.number)


class SubjectIndex:
    def __init__(self, courses: Iterable, names: Mapping[str, str], former: Mapping[str, str]) -> None:
        """courses: CatalogCourse objects (credit ones are searched); names: department code -> subject name;
        former: old code key -> current code (read live, so former codes added after load still match)."""
        self.former = former
        self.entries: list[_Entry] = []
        for c in courses:
            if c.type != "credit":
                continue
            dept = (c.department or c.code.partition(" ")[0]).upper()
            self.entries.append(_Entry(c, code_key(c.code), dept, c.code.partition(" ")[2].upper() or c.code.upper(),
                                       tuple(_words(c.title))))
        self.entries.sort(key=lambda e: (e.department, e.order))
        counts: dict[str, int] = {}
        for e in self.entries:
            counts[e.department] = counts.get(e.department, 0) + 1
        self.subjects: dict[str, Subject] = {}
        for dept, n in counts.items():
            name = names.get(dept)
            aliases = {_flat(dept)}
            if name:
                aliases |= {_flat(name), *(_flat(a) for a in SUBJECT_ALIASES.get(name.lower(), ()))}
            self.subjects[dept] = Subject(dept, name, n, frozenset(aliases))

    # --- Level 1 ------------------------------------------------------------------------------------------
    def resolve_subjects(self, letters: str) -> list[tuple[int, Subject]]:
        """Subjects the letters refer to, best first. A clear hit (tier <= 3) drops the vaguer ones."""
        found = [(t, s) for s in self.subjects.values() if (t := s.match_tier(letters)) is not None]
        if found and min(t for t, _ in found) <= CLEAR_TIER:
            found = [(t, s) for t, s in found if t <= CLEAR_TIER]
        found.sort(key=lambda f: (f[0], len(f[1].label), -f[1].count, f[1].label))
        return found

    # --- Level 2 ------------------------------------------------------------------------------------------
    def search(self, query: str, limit: int = 20) -> dict:
        q = code_key(query)
        if not q:
            return self._result([], 0)
        letters, number = parse_query(query)
        subjects = self.resolve_subjects(letters) if letters else []
        rank = {s.code: i for i, (_, s) in enumerate(subjects)}
        clear = bool(subjects) and subjects[0][0] <= CLEAR_TIER
        words = _words(query)
        former = code_key(self.former.get(q))

        scored: list[tuple[tuple, _Entry, str]] = []
        for e in self.entries:
            if e.department in rank and (number is None or e.number.startswith(number)):
                score, how = (1, rank[e.department], e.number != number, e.order), \
                    "code" if e.number == number else "subject"
            elif e.key == q:
                score, how = (0, e.order), "code"                  # exact code: above any subject listing
            elif former and e.key == former:
                score, how = (0, e.order), "formerly"
            elif clear:
                continue
            elif e.key.startswith(q) or e.number.startswith(q):
                score, how = (3, not e.key.startswith(q), e.number != q, e.department, e.order), "course code"
            elif _prefix_match(words, e.title_words):
                score, how = (4, not e.title_words[0].startswith(words[0]), e.department, e.order), "title"
            else:
                continue
            scored.append((score, e, how))
        scored.sort(key=lambda s: s[0])

        # several subjects share the list: each gets a fair share before the next one fills it
        cap = limit if len(subjects) <= 1 else max(MULTI_SUBJECT_MIN, limit // len(subjects))
        shown: list[tuple[_Entry, str]] = []
        per_subject: dict[str, int] = {}
        for _, e, how in scored:
            if len(shown) == limit:
                break
            if how == "subject":
                per_subject[e.department] = per_subject.get(e.department, 0) + 1
                if per_subject[e.department] > cap:
                    continue
            shown.append((e, how))
        return self._result(shown, len(scored), {s.code: t for t, s in subjects})

    def search_in_subject(self, code: str, query: str, limit: int = 100) -> dict | None:
        """One subject only (None if the catalog has no such subject). Letters that name the subject itself are
        ignored before a number ('math 1' in MTH == '1'); other letters search its course titles."""
        subject = self.subjects.get(code_key(code))
        if subject is None:
            return None
        entries = [e for e in self.entries if e.department == subject.code]
        letters, number = parse_query(query)
        names_subject = bool(letters) and (subject.match_tier(letters) or 99) <= CLEAR_TIER
        if names_subject and number:
            letters = ""
        letter_words, all_words = _words(letters), _words(query)

        scored: list[tuple[tuple, _Entry, str]] = []
        for e in entries:
            if not letters and not number:
                score, how = (0, e.order), "subject"
            elif not letters:
                if not e.number.startswith(number):
                    continue
                score, how = (0 if e.number == number else 1, e.order), "code" if e.number == number else "course code"
            elif _prefix_match(all_words, e.title_words) or (
                    number and e.number.startswith(number) and _prefix_match(letter_words, e.title_words)):
                score, how = (2, not e.title_words[0].startswith(letter_words[0]), e.order), "title"
            else:
                continue
            scored.append((score, e, how))
        if not scored and names_subject:                # 'math' inside Mathematics: no title hit -> the subject
            scored = [((0, e.order), e, "subject") for e in entries]
        scored.sort(key=lambda s: s[0])
        shown = [(e, how) for _, e, how in scored[:limit]]
        return {**self._result(shown, len(scored)), "subject_filter": subject.summary()}

    # --- output -------------------------------------------------------------------------------------------
    def _result(self, shown: list[tuple[_Entry, str]], total: int, tiers: dict[str, int] | None = None) -> dict:
        """Courses grouped by subject (best group first) + one summary per group, for the dropdown's headers.
        `match` says how the typed letters named that subject (None: its courses matched by code or title)."""
        tiers = tiers or {}
        group_order: dict[str, int] = {}
        for e, _ in shown:
            group_order.setdefault(e.department, len(group_order))
        shown = sorted(shown, key=lambda r: group_order[r[0].department])        # stable: keeps rank inside a group
        return {
            "total": total,
            "subject_filter": None,
            "subjects": [{**self.subjects[d].summary(), "match": SUBJECT_MATCH.get(tiers.get(d, 0))}
                         for d in group_order],
            "courses": [{**e.course.compact(), "subject": e.department, "matched_by": how} for e, how in shown],
        }
