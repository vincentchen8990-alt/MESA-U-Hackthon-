"""
/generate-sep for loaded pathways: joins the separate data layers, then hands sep_engine one
concrete problem.

    planner input
    -> canonical institutions / major / academic year        (PathwayStore lookup, done by main.py)
    -> matching articulation agreement                        (its valid ALTERNATIVES; OR stays OR)
    -> completed courses, canonicalized                        (catalog codes, former codes, cross-listings)
    -> community-college AP waivers                            (only what the college's AP chart states)
    -> remaining major preparation + prerequisites             (prerequisite registry, non-guessing resolver)
    -> remaining GE                                            (the college's own Cal-GETC list for that year)
       CSU Golden Four: only Cal-GETC 1A/1B/1C/2 from that list, then just enough other GE for the CSU
       30-GE-unit minimum, then elective slots for the 60-transferable-unit minimum (see _choose)
    -> major-prep / GE overlap                                 (GE slots compete with major courses in Dijkstra)
    -> schedule (sep_engine.generate_sep)
    -> structured plan: typed courses, GE slots, remaining_after_transfer, data sources, AP layers, notes

Nothing here invents articulation, prerequisites, GE eligibility, AP equivalencies or data years: whatever the
data can't support is returned as needs_review / not_loaded / unknown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from articulation.assist import Alternative, _dnf, to_engine
from datastores import DataRegistry, university_system
from datastores.common import academic_year_of, code_key, split_code, year_status
from datastores.ge import (CAL_GETC, COURSE_LIST_OF, CSU_GOLDEN_FOUR, PATHWAY_LABELS, UC_SEVEN, CollegeGE, GEItem,
                           SlotSpec, course_item, match_items, parse_ge_areas, pathway_id)
from datastores.prerequisites import PrereqRecord, PrereqResolver, ResolvedPrereqs
from sep_engine import (Course, GroupCategory, RequirementGroup, SEPError, StudentProfile, TransferAgreement,
                        generate_sep, select_pathway)
from sep_engine.pipeline import node_id
from sep_engine.scheduler import parse_term_label

TOP_ALTERNATIVES = 12
PLACEHOLDER_DIFFICULTY = 1.0      # a GE slot wins a units tie against a real course picked only for GE
GOLDEN_FOUR_EARLY = 8.0           # Course.early for Golden Four slots: a soft pull toward the first terms
ELECTIVE_UNITS = 3.0              # a typical "CSU-transferable elective" slot (no course is named)
ELECTIVE_GROUP = "csu-electives"


class PlanError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class AltSpec:
    index: int
    required: list[str]
    groups: list[RequirementGroup]
    remaining: list[str]                  # requirement ids (ASSIST) or labels (legacy)
    path: list[str]
    alt: Alternative | None = None


@dataclass
class SlotReq:
    spec: SlotSpec
    lab_mode: str | None                  # None | 'with_lab' | 'lab_only'
    role: str | None = None               # CSU Golden Four mode: 'golden_four' | 'csu_ge_units'

    @property
    def group_id(self) -> str:
        return f"ge-{self.spec.slot_id}" + ("-lab" if self.lab_mode == "with_lab" else "")


@dataclass
class Built:
    catalog: dict[str, Course]
    agreement: TransferAgreement
    completed: list[str]
    placeholders: dict[str, dict]         # placeholder code -> GE slot metadata
    resolved: ResolvedPrereqs
    alt: AltSpec
    variant: list[SlotReq]
    ge_links: list[tuple[str, str]] = field(default_factory=list)   # (GE slot code, course it is planned after)
    electives: list[str] = field(default_factory=list)              # elective slot codes (CSU Golden Four mode)
    selection: Any = None                                           # the Dijkstra selection _choose scored


@dataclass
class GEPrereqs:
    """The edges GE slots get from the prerequisites of the courses that can fill them (Planner._ge_prereqs)."""
    prereqs: dict[str, list[str]] = field(default_factory=dict)     # slot / supporting course -> earlier-term codes
    coreqs: dict[str, list[str]] = field(default_factory=dict)      # slot / supporting course -> same term or earlier
    added: dict[str, str] = field(default_factory=dict)             # course added only to enable a slot -> slot
    links: list[tuple[str, str]] = field(default_factory=list)      # (slot code, course it is planned after)
    parts: list[ResolvedPrereqs] = field(default_factory=list)

    def courses(self) -> list[str]:
        """Every real course these edges name (slots excluded): what the plan's catalog must contain."""
        return [c for deps in (*self.prereqs.values(), *self.coreqs.values()) for c in deps]

    def merged_into(self, base: ResolvedPrereqs) -> ResolvedPrereqs:
        """`base` (the major prep's resolution, shared between builds) plus these, as a new object."""
        out = ResolvedPrereqs(prereqs=dict(base.prereqs), coreqs=dict(base.coreqs), added=dict(base.added),
                              reviews=list(base.reviews), sources=dict(base.sources), statuses=dict(base.statuses),
                              unconfirmed_year=list(base.unconfirmed_year))
        for part in self.parts:
            for code, deps in part.prereqs.items():
                out.prereqs.setdefault(code, deps)
            for code, deps in part.coreqs.items():
                out.coreqs.setdefault(code, deps)
            for k, v in part.sources.items():
                out.sources.setdefault(k, v)
            for k, v in part.statuses.items():
                out.statuses.setdefault(k, v)
            out.unconfirmed_year += [c for c in part.unconfirmed_year if c not in out.unconfirmed_year]
            seen = {(r.course, r.kind, r.message) for r in out.reviews}
            out.reviews += [r for r in part.reviews if (r.course, r.kind, r.message) not in seen]
        for code, why in self.added.items():
            out.added.setdefault(code, why)
        return out


@dataclass
class Note:
    kind: str
    message: str
    course: str | None = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "message": self.message, **({"course": self.course} if self.course else {})}


@dataclass
class Planner:
    req: Any
    match: Any
    data: DataRegistry
    notes: list[Note] = field(default_factory=list)
    golden_four: Any = None               # CSU Golden Four rules (datastores.ge.CSUGoldenFour), set by run()

    def note(self, kind: str, message: str, course: str | None = None) -> None:
        if not any(n.message == message for n in self.notes):
            self.notes.append(Note(kind, message, course))

    # =================================================================
    # entry point
    # =================================================================
    def run(self) -> dict:
        p = self.pathway = self.match.pathway
        self.ag = p.assist
        self.college_id, self.university_id = p.college_ref, p.university_ref
        self.wanted = academic_year_of(self.req.start_term)
        self.system = university_system(p.university)
        self.ge_pid = getattr(self.req, "transfer_pathway", None) or pathway_id(self.req.ge_pathway)
        if self.ge_pid == UC_SEVEN and self.system != "UC":
            raise PlanError(422, f"The UC 7-course pattern is a UC transfer admission requirement, so it doesn't apply "
                                 f"to {p.university}. Choose CAL-GETC for this target.")
        if self.ge_pid == CSU_GOLDEN_FOUR and self.system != "CSU":
            raise PlanError(422, f"The CSU Golden Four is a CSU transfer admission minimum, so it doesn't apply to "
                                 f"{p.university}. Choose CAL-GETC for this target.")
        self.golden_four = self.data.ge.golden_four if self.ge_pid == CSU_GOLDEN_FOUR else None
        for w in self.match.warnings:
            self.note("articulation", w)
        self.catalog = self.data.catalogs.for_institution(self.college_id, self.wanted)
        self.prereg = self.data.prereqs.get(self.college_id)
        self.ge_ds: CollegeGE | None = (self.data.ge.college_list(self.college_id, COURSE_LIST_OF[self.ge_pid],
                                                                  self.wanted)
                                        if self.ge_pid in COURSE_LIST_OF else None)
        self._completed()
        self._ap()
        self._ge_state()
        built = self._choose()
        season, year = parse_term_label(self.req.start_term)
        profile = StudentProfile(completed_courses=built.completed, start_term=season, start_year=year,
                                 include_summer=self.req.include_summer)
        try:
            result = generate_sep(built.catalog, built.agreement, profile)
        except SEPError as e:
            raise PlanError(422, str(e)) from e
        for w in result.pop("warnings"):
            self.note("engine", w)
        result["pathway"].update(transfer_pathway=self.ge_pid, ge_pathway=PATHWAY_LABELS[self.ge_pid])
        for plan in result["plans"]:
            self._annotate(plan, built)
        in_plan = {c["code"] for plan in result["plans"] for s in plan["semesters"] for c in s["courses"]}
        if self.ge_pid == CSU_GOLDEN_FOUR:
            self._golden_four_notes(result["plans"])
        self._prereq_notes(built.resolved, in_plan)
        self._ge_order_notes(built, in_plan)
        self._source_notes()
        result["articulation"] = self._articulation_info(built, result["plans"][0] if result["plans"] else None)
        result["remaining_after_transfer"] = self._remaining(built.alt)
        if result["remaining_after_transfer"] and self.ag:   # legacy files already say so in their notes
            items = "; ".join(r["label"] for r in result["remaining_after_transfer"])
            self.note("remaining_after_transfer", (
                f"{len(result['remaining_after_transfer'])} {p.university} requirement(s) have no {p.college} course "
                f"articulated, so they are completed after transfer: {items}."))
        result["data_sources"] = self._data_sources()
        result["ap_evaluations"] = self.ap_evals
        result["completed_courses_normalized"] = self.completed_rows
        result["admission_requirements"] = self._admission_requirements()
        result["review_items"] = [{"course": r.course, "kind": r.kind, "message": r.message}
                                  for r in built.resolved.reviews if r.course in in_plan]
        result["notes"] = [n.to_dict() for n in self.notes]
        result["warnings"] = [n.message for n in self.notes]
        return result

    # =================================================================
    # completed courses
    # =================================================================
    def _course_index(self) -> dict[str, tuple[str, str, float | None]]:
        """Courses the agreement itself describes: key -> (code, title, units)."""
        if self.ag:
            return {c.key: (c.code, c.title, c.units) for c in self.ag.courses.values()}
        return {code_key(c.code): (c.code, c.title, c.units) for c in self.pathway.catalog.values()}

    def _completed(self) -> None:
        index = self._course_index()
        cross = self.ag.cross_listed() if self.ag else {}
        rows, codes = [], []
        for raw in self.req.completed_courses:
            k = code_key(raw)
            if not k:
                continue
            code, by = None, "unknown"
            if k in cross and cross[k] in index:
                code, by = index[cross[k]][0], "cross_listed"
            elif k in index:
                code, by = index[k][0], "agreement"
            elif self.catalog and (hit := self.catalog.resolve(raw)):
                code, by = hit[0].code, hit[1]
            elif self.ge_ds and (g := self.ge_ds.course(raw)):
                code, by = g.code, "ge_list"
            rows.append({"entered": raw, "code": code, "matched_by": by})
            if code and code not in codes:
                codes.append(code)
        unknown = [r["entered"] for r in rows if not r["code"]]
        if unknown:
            where = f"the {self.catalog.title}" if self.catalog else f"the loaded {self.pathway.college} data"
            self.note("completed_courses", f"Not found in {where}, so not counted: {', '.join(unknown)}.")
        self.completed_rows = rows
        self.completed_real = codes              # entered by the student (canonical codes)

    # =================================================================
    # AP: community college effect, GE effect, target campus effect (kept apart)
    # =================================================================
    def _ap(self) -> None:
        ap = self.data.ap
        relevant = set(self._course_index())
        self.ap_evals, self.ap_waived, self.ap_ge_items = [], [], []
        uc = self.data.ge.uc_seven
        for entry in self.req.ap_scores:
            subject, score = entry.subject, entry.score
            exam_id = ap.normalize_exam(subject)
            if exam_id is None:
                self.ap_evals.append({"subject": subject, "score": score, "exam_id": None, "status": "unknown_exam",
                                      "message": "This AP subject isn't in the loaded AP exam list."})
                continue
            cc = ap.community_college_effect(self.college_id, exam_id, score, relevant)
            for c in cc.get("waived_courses") or []:
                if c not in self.ap_waived:
                    self.ap_waived.append(c)
            for cond in cc.get("conditions") or []:
                if cc["status"] == "applied":
                    self.note("ap", f"{ap.exams.get(exam_id, subject)}: {cond}")
            ge = self._ap_ge_effect(exam_id, score, cc, uc)
            target = ap.target_campus_effect(self.university_id, exam_id, score)
            self.ap_evals.append({"subject": subject, "exam_id": exam_id, "exam_name": ap.exams.get(exam_id, subject),
                                  "score": score, "status": "evaluated", "community_college_effect": cc,
                                  "ge_effect": ge, "target_campus_effect": target})
        if any(e.get("target_campus_effect", {}).get("rules") for e in self.ap_evals):
            self.note("ap", f"AP credit at {self.pathway.university} is listed with each exam for reference. The campus "
                            f"decides it (it needs review), so it isn't applied to your {self.pathway.college} plan.")

    def _ap_ge_effect(self, exam_id: str, score: int, cc: dict, uc) -> dict:
        if self.ge_pid == UC_SEVEN:
            if uc is None:
                return {"pathway": UC_SEVEN, "status": "not_loaded", "areas": []}
            areas = uc.ap_exams.get(exam_id)
            if not areas:
                return {"pathway": UC_SEVEN, "status": "not_listed", "areas": []}
            if score < uc.ap_min_score:
                return {"pathway": UC_SEVEN, "status": "below_minimum", "areas": areas}
            reqs = uc.requirements
            options = tuple((r["id"], False) for a in areas for r in reqs if a in (r.get("accepts") or []))
            self.ap_ge_items.append(GEItem("ap_exam", exam_id, f"{self.data.ap.exams.get(exam_id, exam_id)} ({score})",
                                           areas[0], options))
            return {"pathway": UC_SEVEN, "status": "eligible", "areas": areas, "source": uc.source.get("document_title")}
        if self.ge_ds is None:
            return {"pathway": self.ge_pid, "status": "not_loaded", "areas": []}
        text, pid = cc.get("ge_areas_text"), self.ge_pid
        if cc["status"] in ("not_loaded", "not_listed"):
            return {"pathway": pid, "status": cc["status"], "areas": []}
        if cc["status"] == "below_minimum":
            return {"pathway": pid, "status": "below_minimum", "areas_text": text, "areas": []}
        if not text:
            return {"pathway": pid, "status": "none_listed", "areas": []}
        options = parse_ge_areas(text, self.ge_ds.lab["subarea_id"] if self.ge_ds.lab else None)
        if not options:
            self.note("ap", f"{self.data.ap.exams.get(exam_id)}: the college chart's GE entry ({text}) couldn't be read, "
                            f"so it wasn't applied; ask a counselor.")
            return {"pathway": pid, "status": "needs_review", "areas_text": text, "areas": []}
        discipline = (cc.get("listed_courses") or [exam_id])[0].split()[0]
        self.ap_ge_items.append(GEItem("ap_exam", exam_id, f"{self.data.ap.exams.get(exam_id, exam_id)} ({score})",
                                       discipline, options))
        return {"pathway": pid, "status": "eligible", "areas_text": text,
                "areas": [a for a, _ in options if a], "choose_one": " or " in text.split(";")[0],
                "lab": any(lab for _, lab in options), "source": cc.get("source")}

    # =================================================================
    # GE: what completed courses and AP already cover; slot variants for the rest
    # =================================================================
    def _ge_state(self) -> None:
        self.ge_states, self.ge_lab_item, self.ge_variants, self.ge_label = [], None, [[]], None
        self.csu_other_specs, self.csu_other_states = [], []
        if self.ge_pid == CSU_GOLDEN_FOUR:
            self._golden_four_state()
        elif self.ge_pid == CAL_GETC:
            ds = self.ge_ds
            if ds is None:
                years = self.data.ge.years(self.college_id, CAL_GETC)
                self.note("ge", (f"GE isn't planned: no Cal-GETC course list for {self.pathway.college} is loaded"
                                 + (f" for {self.wanted} or an earlier year (loaded: {', '.join(years)})." if years else ".")))
                return
            self.ge_label = ds.display_name
            items = [it for c in self.completed_real if c not in self.ap_waived
                     for it in [course_item(ds, c)] if it]
            items += self.ap_ge_items
            self.ge_states, self.ge_lab_item = match_items(ds.slots, bool(ds.lab), items)
            remaining = [s.spec for s in self.ge_states if s.satisfied_by is None]
            lab_needed = bool(ds.lab) and self.ge_lab_item is None
            if not lab_needed:
                self.ge_variants = [[SlotReq(s, None) for s in remaining]]
            else:
                lab_slots = [s for s in remaining if s.lab_area]
                if lab_slots:
                    self.ge_variants = [[SlotReq(s, "with_lab" if s is ls else None) for s in remaining] for ls in lab_slots]
                else:
                    lab = ds.lab
                    spec = SlotSpec(lab["subarea_id"], lab["subarea_id"], lab["area_id"], lab["name"],
                                    lab["name"], lab["eligible"], None, None, True)
                    self.ge_variants = [[SlotReq(s, None) for s in remaining] + [SlotReq(spec, "lab_only")]]
        elif self.ge_pid == UC_SEVEN:
            uc = self.data.ge.uc_seven
            if uc is None:
                self.note("ge", "The UC 7-course pattern isn't loaded, so it isn't planned.")
                return
            self.ge_label = "UC 7-course pattern"
            specs = []
            for r in uc.requirements:
                n = int(r.get("minimum_courses") or 1)
                for i in range(n):
                    specs.append(SlotSpec(r["id"] if n == 1 else f"{r['id']}#{i + 1}", r["id"], r["id"], r["name"],
                                          r.get("area_name") or r["name"], frozenset(), r.get("rule"),
                                          r.get("minimum_semester_units_each"), False,
                                          int(r.get("minimum_distinct_areas") or 0)))
            self.uc_specs = specs
            self.ge_states, _ = match_items(specs, False, self.ap_ge_items)
            self.ge_variants = [[SlotReq(s.spec, None) for s in self.ge_states if s.satisfied_by is None]]
            self.note("ge", ("The UC 7-course pattern file has no college course list, so completed courses and "
                             "major-prep courses can't be checked against it; each remaining course is planned as its "
                             "own slot. Confirm UC-E/UC-M/UC-H/UC-B/UC-S courses on ASSIST."))

    def _golden_four_state(self) -> None:
        """
        CSU Golden Four: the four areas are the college's own Cal-GETC 1A, 1B, 1C and 2 slots (course lists from
        that list, never a list of their own). Completed courses and AP (as the college's AP chart states) are
        matched to the four areas FIRST, then what is left to the other Cal-GETC areas: those only count toward
        the CSU 30-GE-unit minimum. Only the unfilled Golden Four areas become slots here; _choose adds other GE
        and electives only as far as the CSU unit minimums need them.
        """
        ds, g4 = self.ge_ds, self.golden_four
        if g4 is None:
            self.note("ge", "The CSU Golden Four rules file isn't loaded, so the Golden Four isn't planned.")
            return
        if ds is None:
            years = self.data.ge.years(self.college_id, CAL_GETC)
            self.note("ge", (f"The Golden Four isn't planned: its courses come from {self.pathway.college}'s Cal-GETC "
                             f"list, and none is loaded" + (f" for {self.wanted} or an earlier year (loaded: "
                                                            f"{', '.join(years)})." if years else ".")))
            return
        self.ge_label = g4.display_name
        ids = set(g4.area_ids)
        g4_specs = [s for s in ds.slots if s.requirement_id in ids]
        missing = [a for a in g4.area_ids if not any(s.requirement_id == a for s in g4_specs)]
        if missing:
            self.note("ge", f"{ds.institution_name}'s {ds.display_name} list has no area {', '.join(missing)}, so that "
                            f"Golden Four area isn't planned; check with a counselor.")
        self.csu_other_specs = [s for s in ds.slots if s.requirement_id not in ids]
        items = [it for c in self.completed_real if c not in self.ap_waived for it in [course_item(ds, c)] if it]
        items += self.ap_ge_items
        self.ge_states, _ = match_items(g4_specs, False, items)
        used = [st.satisfied_by for st in self.ge_states if st.satisfied_by]
        self.csu_other_states, _ = match_items(self.csu_other_specs, False, [i for i in items if i not in used])
        self.ge_variants = [[SlotReq(st.spec, None, "golden_four") for st in self.ge_states if st.satisfied_by is None]]

    # =================================================================
    # alternatives x GE variants -> the cheapest valid engine problem
    # =================================================================
    def _alternatives(self) -> list[AltSpec]:
        if self.ag is None:
            ag = self.pathway.agreement
            return [AltSpec(0, list(ag.required_courses), [g for g in ag.requirement_groups
                                                            if g.category is GroupCategory.MAJOR],
                            list(self.pathway.not_articulated), [])]
        out = []
        for i, alt in enumerate(self.ag.alternatives):
            required, groups = to_engine(self.ag, alt)
            out.append(AltSpec(i, required, groups, list(alt.remaining), list(alt.path), alt))
        done = {code_key(c) for c in self.completed_real + self.ap_waived}
        index = self._course_index()

        def bound(a: AltSpec) -> tuple:
            units = sum(index[code_key(c)][2] or 0 for c in a.required if code_key(c) not in done)
            units += sum(min((index[code_key(o)][2] or 0) for o in g.options) for g in a.groups
                         if not any(code_key(o) in done for o in g.options))
            return (len(a.remaining), units, a.index)

        return sorted(out, key=bound)[:TOP_ALTERNATIVES]

    def _choose(self) -> Built:
        self.resolved_cache: dict[int, ResolvedPrereqs] = {}
        best = self._choose_among(self.ge_variants)
        if self.ge_pid != CSU_GOLDEN_FOUR or not self.ge_ds or not self.golden_four:
            return best
        # CSU minimums, in order: the Golden Four (the variant), then GE-level units, then transferable units.
        # Each extra GE slot is a real Cal-GETC area of the college's list that nothing in the plan fills yet, so a
        # major-prep course on the list (PHYS 4A -> 5A) counts first and no GE is added beyond the minimum.
        for _ in range(len(self.csu_other_specs) + 1):
            tally = self._csu_tally(best, best.selection.choices, best.selection.courses)
            gap = self.golden_four.ge_units - tally["ge_units"]["total"]
            extra = self._extra_ge_slots(best, tally, gap) if gap > 1e-9 else []
            if not extra:
                break
            best = self._choose_among([[*best.variant, *extra]])
        tally = self._csu_tally(best, best.selection.choices, best.selection.courses)
        short = self.golden_four.transferable_units - tally["transferable_units"]["total"]
        if short > 1e-9:
            # No reliable data names which other courses are CSU-transferable: generic elective slots, no course
            built = self._build(best.alt, best.variant, self.resolved_cache, electives=_elective_units(short))
            built.selection = select_pathway(built.catalog, built.agreement, built.completed, "units")
            best = built
        # A real course that fills a Golden Four area (MTH 1 -> 2) goes early like the Golden Four slots do.
        # Course.early only affects term placement, never which courses are selected.
        for slot in best.variant:
            if slot.role == "golden_four":
                for code in best.selection.choices.get(slot.group_id, []):
                    if code not in best.placeholders:
                        best.catalog[code] = best.catalog[code].model_copy(update={"early": GOLDEN_FOUR_EARLY})
        return best

    def _choose_among(self, variants: list[list[SlotReq]]) -> Built:
        errors, best = [], None
        for alt in self._alternatives():
            for variant in variants:
                try:
                    built = self._build(alt, variant, self.resolved_cache)
                    sel = select_pathway(built.catalog, built.agreement, built.completed, "units")
                except SEPError as e:
                    errors.append(str(e))
                    continue
                built.selection = sel
                key = (len(alt.remaining), sel.cost, alt.index)
                if best is None or key < best[0]:
                    best = (key, built)
        if best is None:
            raise PlanError(422, "No valid plan could be built from the loaded data: " + (errors[0] if errors else
                                                                                          "no alternatives"))
        return best[1]

    def _extra_ge_slots(self, built: Built, tally: dict, gap: float) -> list[SlotReq]:
        """Other Cal-GETC areas for the CSU 30-GE-unit minimum: only areas nothing already fills (completed, AP,
        a planned course on the list), one course per area before a second course of the same area, in the list's
        order, until `gap` units are covered. They count toward CSU GE units, not toward the Golden Four."""
        taken = {s.spec.slot_id for s in built.variant} | tally["ge_units"]["covered_slots"]
        free = [s for s in self.csu_other_specs if s.slot_id not in taken]
        area_order = list(dict.fromkeys(s.area_id for s in self.csu_other_specs))
        nth = {s.slot_id: [x.slot_id for x in self.csu_other_specs if x.area_id == s.area_id].index(s.slot_id)
               for s in free}
        out, units = [], 0.0
        for spec in sorted(free, key=lambda s: (nth[s.slot_id], area_order.index(s.area_id))):
            u = self.ge_ds.requirement_units(spec.eligible)
            if u is None:
                continue
            out.append(SlotReq(spec, None, "csu_ge_units"))
            units += u
            if units >= gap - 1e-9:
                break
        return out

    def _units_of(self, key: str) -> tuple[str, str, float | None] | None:
        index = self._course_index()
        if key in index and index[key][2]:
            return index[key]
        if self.catalog:
            c = self.catalog.get(key)
            if c and c.type == "credit" and c.units:
                return (c.code, c.title, c.units)
        return None

    def _record(self, key: str) -> PrereqRecord | None:
        if self.prereg and key in self.prereg.records:
            return self.prereg.records[key]
        if self.ag and key in self.ag.prereq_records:
            return self.ag.prereq_records[key]
        if self.ag is None:                                   # legacy file: its own prereqs/coreqs lists
            course = next((c for c in self.pathway.catalog.values() if code_key(c.code) == key), None)
            if course is not None:
                items = ([{"type": "COURSE", "course_id": f"x:{code_key(p)}", "timing": "before"} for p in course.prereqs]
                         + [{"type": "COURSE", "course_id": f"x:{code_key(q)}", "timing": "before_or_concurrent"}
                            for q in course.coreqs])
                return PrereqRecord(key, course.code, "documented", {"type": "AND", "items": items} if items
                                    else {"type": "NONE"}, None, (), None, f"agreement:{self.pathway.file}")
        return None

    def _build(self, alt: AltSpec, variant: list[SlotReq], cache: dict[int, ResolvedPrereqs],
               electives: list[float] = ()) -> Built:
        completed_all = [*self.completed_real, *(c for c in self.ap_waived if c not in self.completed_real)]
        seeds = list(dict.fromkeys([*alt.required, *(o for g in alt.groups for o in g.options)]))
        if alt.index not in cache:
            resolver = PrereqResolver(self._record, self._units_of, completed_all,
                                      self.prereg.alias_of if self.prereg else None,
                                      self.prereg.external if self.prereg else None,
                                      self.prereg.policy_satisfied if self.prereg else None)
            cache[alt.index] = resolver.resolve(seeds, anchor=alt.required)
        resolved = cache[alt.index]
        codes = list(dict.fromkeys([*seeds, *resolved.added,
                                    *(d for deps in (*resolved.prereqs.values(), *resolved.coreqs.values()) for d in deps)]))
        catalog: dict[str, Course] = {}
        for code in codes:
            info = self._units_of(code_key(code))
            if info is None:
                raise SEPError(f"{code} has no units in the loaded data, so it can't be planned.")
            catalog[info[0]] = None                       # placeholder: filled below once every code is known
        for code in list(catalog):
            info = self._units_of(code_key(code))
            catalog[code] = Course(code=code, title=info[1], units=info[2],
                                   prereqs=[p for p in resolved.prereqs.get(code, []) if p in catalog and p != code],
                                   coreqs=[q for q in resolved.coreqs.get(code, []) if q in catalog and q != code])
        # GE slots are placeholders, but the courses that fill them can have prerequisites: give each slot the
        # edges that make at least one of its approved courses takeable in its term (see _ge_prereqs)
        ge = self._ge_prereqs(variant, catalog, completed_all)
        new = [c for c in dict.fromkeys(ge.courses()) if c not in catalog and self._units_of(code_key(c)) is not None]
        golden = frozenset().union(*(self._slot_eligible(s) for s in variant if s.role == "golden_four"))
        for code in new:                                 # courses a GE slot's approved courses need (e.g. ENGL C1000)
            info = self._units_of(code_key(code))
            catalog[code] = Course(code=code, title=info[1], units=info[2],
                                   prereqs=[p for p in ge.prereqs.get(code, []) if p in catalog or p in new],
                                   coreqs=[q for q in ge.coreqs.get(code, []) if q in catalog or q in new],
                                   early=GOLDEN_FOUR_EARLY if code_key(code) in golden else 0.0)
        completed = [c for c in completed_all if c in catalog]
        placeholders, ge_groups = {}, []
        for slot in variant:
            ph = self._placeholder(slot, catalog)
            if ph is None:
                continue
            course, meta, candidates = ph
            before = [p for p in ge.prereqs.get(course.code, []) if p in catalog]
            with_or_before = [q for q in ge.coreqs.get(course.code, []) if q in catalog]
            if before or with_or_before:
                course = Course(code=course.code, title=course.title, units=course.units, difficulty=course.difficulty,
                                prereqs=before, coreqs=with_or_before, early=course.early)
                meta["planned_after"] = [*before, *with_or_before]
            catalog[course.code] = course
            placeholders[course.code] = meta
            ge_groups.append(RequirementGroup(id=slot.group_id, name=meta["label"], category=GroupCategory.GE,
                                              choose=1, options=[*candidates, course.code]))
        elective_codes = [f"ELECTIVE {i + 1}" for i in range(len(electives))]
        for code, units in zip(elective_codes, electives):   # CSU Golden Four: units toward 60, no course named
            catalog[code] = Course(code=code, title="CSU-transferable elective", units=units,
                                   difficulty=PLACEHOLDER_DIFFICULTY)
        if elective_codes:
            ge_groups.append(RequirementGroup(id=ELECTIVE_GROUP, name="Additional CSU-transferable electives",
                                              category=GroupCategory.GE, choose=len(elective_codes),
                                              options=elective_codes))
        resolved = ge.merged_into(resolved)
        p = self.pathway
        agreement = TransferAgreement(
            college=p.college, university=p.university, major=p.major,
            degree=(f"{p.major}, {p.degree}" if p.degree else None), ge_pattern=self.req.ge_pathway,
            required_courses=alt.required, requirement_groups=[*alt.groups, *ge_groups])
        return Built(catalog, agreement, completed, placeholders, resolved, alt, variant, ge.links, elective_codes)

    def _slot_eligible(self, slot: SlotReq) -> frozenset[str]:
        """Code keys of the courses that can fill a GE slot (none for the UC 7-course pattern: no course list)."""
        if self.ge_pid == UC_SEVEN or not self.ge_ds:
            return frozenset()
        spec = slot.spec
        return spec.eligible & self.ge_ds.lab["eligible"] if slot.lab_mode == "with_lab" else spec.eligible

    @staticmethod
    def _slot_code(slot: SlotReq) -> str:
        return f"GE {slot.spec.slot_id}" + ("+LAB" if slot.lab_mode == "with_lab" else "")

    # =================================================================
    # GE prerequisites: a slot is only placed where one of its approved courses can be taken
    # =================================================================
    def _ge_rule(self, slot: SlotReq) -> dict | None:
        """A GE slot's prerequisite rule: ONE of its approved courses must be takeable, so it is an OR over each
        course's own rule (kept whole, never merged into one AND). None: an approved course has no prerequisite
        (the registry says NONE, or the GE list shows none and the registry has no record), so any term works.
        A course whose prerequisite isn't in the structured data becomes a CONDITION branch: not assumed met, and
        never turned into a guessed course edge."""
        branches = []
        for key in sorted(self._slot_eligible(slot)):
            rec = self._record(key)
            if rec is not None and rec.researched:
                if rec.requirement.get("type") == "NONE":
                    return None
                branches.append(rec.requirement)
                continue
            listed = self.ge_ds.courses.get(key)
            text = (listed.prerequisite if listed else None) or (rec.summary if rec else None)
            if rec is None and not text:
                return None
            branches.append({"type": "CONDITION", "description": (
                f"{listed.code if listed else split_code(key)}: {text or 'prerequisite rule unresolved'} "
                f"(not in the structured prerequisite data)")})
        return {"type": "OR", "items": branches} if branches else None

    def _ge_prereqs(self, variant: list[SlotReq], catalog: dict[str, Course], completed_all: list[str]) -> GEPrereqs:
        """
        Edges that make each GE slot's placement valid, from the courses that can fill it:

            slot rule = OR(rule of approved course 1, rule of approved course 2, ...)

        resolved by the same PrereqResolver as major prep (completed, AP waivers, aliases, college policy,
        before / before_or_concurrent / same_term, AND / OR / CONDITION / unresolved). Its "already in the plan"
        set holds the plan's courses AND the courses that can fill the OTHER remaining GE slots: a prerequisite
        that fills another GE requirement costs nothing extra, so e.g. ENGL 4A / ENGL C1001 (Cal-GETC 1B) both
        needing ENGL C1000 puts ENGL C1000 in the plan, where it also fills Cal-GETC 1A, and GE 1B after it. A slot
        with an approved course that needs nothing gets no edge, so GE keeps spreading freely.
        """
        out = GEPrereqs()
        if not self.ge_ds:
            return out
        rules = {self._slot_code(s): (s, rule) for s in variant if (rule := self._ge_rule(s)) is not None}
        if not rules:
            return out
        plan_keys = {code_key(c) for c in catalog}
        fillers = {self._slot_code(s): {k for k in self._slot_eligible(s) if self._units_of(k) is not None}
                   for s in variant}
        filler_keys = set().union(*fillers.values())
        for code, (slot, rule) in rules.items():
            key = code_key(code)
            label = f"{self.ge_label} {slot.spec.requirement_id}: {slot.spec.name}"
            units = self.ge_ds.requirement_units(self._slot_eligible(slot)) or 3
            record = PrereqRecord(key, code, "documented", rule, None, (), None, f"ge:{self.ge_ds.file}")
            others = set().union(*(v for c, v in fillers.items() if c != code))
            resolver = PrereqResolver(lambda k, _k=key, _r=record: _r if k == _k else self._record(k),
                                      lambda k, _k=key, _v=(code, label, units): _v if k == _k else self._units_of(k),
                                      completed_all, self.prereg.alias_of if self.prereg else None,
                                      self.prereg.external if self.prereg else None,
                                      self.prereg.policy_satisfied if self.prereg else None)
            res = resolver.resolve([code], anchor=[*plan_keys, *others])
            out.parts.append(res)
            for c, deps in res.prereqs.items():
                out.prereqs.setdefault(c, []).extend(d for d in deps if d not in out.prereqs[c])
            for c, deps in res.coreqs.items():
                out.coreqs.setdefault(c, []).extend(d for d in deps if d not in out.coreqs[c])
            for dep, why in res.added.items():                  # rule 4: a course only this slot needs
                out.added.setdefault(dep, why)
            out.links += [(code, d) for d in (*res.prereqs.get(code, []), *res.coreqs.get(code, []))
                          if code_key(d) in filler_keys]
        return out

    def _placeholder(self, slot: SlotReq, catalog: dict[str, Course]):
        spec = slot.spec
        if self.ge_pid == UC_SEVEN:
            units, eligible = spec.minimum_units or 3, frozenset()
        else:
            ds = self.ge_ds
            eligible = self._slot_eligible(slot)
            units = ds.requirement_units(eligible)
            if units is None:
                self.note("ge", f"{ds.display_name} {spec.requirement_id}: no eligible course with units is listed, "
                                f"so this requirement isn't planned; check with a counselor.")
                return None
        suffix = {"with_lab": " (with lab)", "lab_only": " (lab)"}.get(slot.lab_mode or "", "")
        display, name, rule, g4 = self.ge_label, spec.name, spec.rule, self.golden_four
        if slot.role == "golden_four":
            name = g4.area_name(spec.requirement_id) or spec.name
            rule = (f"Golden Four ({name}): required for CSU upper-division transfer"
                    + (f", with a grade of {g4.minimum_grade} or better" if g4.minimum_grade else "")
                    + f". Any course on {self.ge_ds.institution_name}'s {self.ge_ds.display_name} "
                      f"{spec.requirement_id} list counts.")
        elif slot.role == "csu_ge_units":
            display = self.ge_ds.display_name
            rule = (f"Counts toward the CSU minimum of {g4.ge_units:g} GE units; not part of the Golden Four. Any "
                    f"course on {self.ge_ds.institution_name}'s {display} {spec.requirement_id} list counts.")
        label = f"{display} {spec.requirement_id}{suffix}: {name}" + (" (CSU GE units)" if slot.role == "csu_ge_units"
                                                                      else "")
        code = self._slot_code(slot)
        meta = {"pathway": self.ge_pid, "display_name": display, "requirement_id": spec.requirement_id,
                "slot_id": spec.slot_id, "area_id": spec.area_id, "name": name, "area_name": spec.area_name,
                "lab_required": slot.lab_mode == "with_lab", "lab_only": slot.lab_mode == "lab_only",
                "rule": rule, "label": label, "institution_id": self.college_id,
                "academic_year": self.ge_ds.academic_year if self.ge_ds else None,
                "options_status": "loaded" if eligible else "not_loaded"}
        if slot.role:
            meta.update(role=slot.role, course_list=self.ge_ds.display_name)
        candidates = [c for c in catalog if code_key(c) in eligible]
        course = Course(code=code, title=label, units=units, difficulty=PLACEHOLDER_DIFFICULTY,
                        early=GOLDEN_FOUR_EARLY if slot.role == "golden_four" else 0.0)
        return course, meta, candidates

    # =================================================================
    # output
    # =================================================================
    def _req_labels(self) -> dict[str, list[str]]:
        """Course code -> the articulation requirements it can meet (from the agreement's own expressions)."""
        out: dict[str, list[str]] = {}
        if not self.ag:
            return out
        for r in self.ag.requirements.values():
            if not r.expr:
                continue
            for s in _dnf(r.expr):
                for k in s:
                    out.setdefault(self.ag.code(k), [])
                    if r.label not in out[self.ag.code(k)]:
                        out[self.ag.code(k)].append(r.label)
        return out

    def _annotate(self, plan: dict, built: Built) -> None:
        labels = self._req_labels()
        chosen_major = {c["code"] for r in plan["requirements"] if r["category"] == "major" for c in r["courses"]}
        ge_names = {g.id: g.name for g in built.agreement.requirement_groups if g.category is GroupCategory.GE}

        def kind(code: str, is_ge: bool) -> str:
            if code in built.electives:
                return "elective_slot"
            if code in built.placeholders:
                return "ge_slot"
            if is_ge:
                return "ge_course"
            return "major_prep" if code in chosen_major else "prerequisite"

        tally = None
        if self.ge_pid == CSU_GOLDEN_FOUR and self.ge_ds and self.golden_four:
            tally = self._csu_tally(built, {r["id"]: [c["code"] for c in r["courses"]] for r in plan["requirements"]},
                                    {c["code"] for s in plan["semesters"] for c in s["courses"]})
        counted = self._ge_counted(plan, built, tally)
        for sem in plan["semesters"]:
            for c in sem["courses"]:
                c["type"] = kind(c["code"], c["is_ge"])
                if c["type"] == "ge_slot":
                    c["ge"] = built.placeholders[c["code"]]
                    c["requirement_id"] = c["ge"]["requirement_id"]
                    c["satisfies"] = [c["ge"]["label"]]
                    continue
                c["ge_satisfies"] = counted.get(c["code"], [])
                if c["type"] == "major_prep" and labels.get(c["code"]):
                    extra = [s for s in c["satisfies"] if s in ge_names.values()]
                    c["satisfies"] = labels[c["code"]] + extra
        for n in plan["graph"]["nodes"]:
            d = n["data"]
            d["type"] = kind(d["label"], d["is_ge"])
        plan["ge"] = self._ge_block(plan, built)
        if tally is not None:
            plan["transfer_requirements"] = self._transfer_requirements(plan, built, tally)

    def _ge_counted(self, plan: dict, built: Built, tally: dict | None = None) -> dict[str, list[dict]]:
        """Real course code -> the GE requirements the selection counted it toward (e.g. MTH 1 -> Cal-GETC Area 2).
        Only what the GE dataset lists: a course's units never make it GE. CSU Golden Four mode: "Golden Four
        Area 2" for the four areas, "Cal-GETC Area 5A" for a course counted only toward the CSU GE units."""
        slot_of = {s.group_id: s for s in built.variant}
        area = "" if self.ge_pid == UC_SEVEN else "Area "
        out: dict[str, list[dict]] = {}

        def add(code: str, requirement_id: str, area_id: str, name: str, display: str | None = None) -> None:
            display = display or self.ge_label
            out.setdefault(code, []).append({
                "pathway": self.ge_pid, "display_name": display, "area": requirement_id, "area_id": area_id,
                "name": name, "label": f"{display} {area}{requirement_id}"})

        for r in plan["requirements"]:
            slot = slot_of.get(r["id"])
            if r["category"] != "ge" or slot is None:
                continue
            for c in r["courses"]:
                if c["code"] in built.placeholders:
                    continue
                spec = slot.spec
                if slot.role == "golden_four":
                    add(c["code"], spec.requirement_id, spec.area_id,
                        self.golden_four.area_name(spec.requirement_id) or spec.name, "Golden Four")
                elif slot.role == "csu_ge_units":
                    add(c["code"], spec.requirement_id, spec.area_id, spec.name, self.ge_ds.display_name)
                else:
                    add(c["code"], spec.requirement_id, spec.area_id, spec.name)
                if slot.lab_mode == "with_lab" and self.ge_ds and self.ge_ds.lab:
                    lab = self.ge_ds.lab
                    add(c["code"], lab["subarea_id"], lab["area_id"], lab["name"])
        for row in (tally or {}).get("ge_units", {}).get("courses", []):
            if row["via"] == "listed_course":                   # e.g. PHYS 4A -> CSU GE units via Cal-GETC 5A
                add(row["code"], row["requirement_id"], row["area_id"], row["name"], self.ge_ds.display_name)
        return out

    def _ge_block(self, plan: dict, built: Built) -> dict | None:
        if not self.ge_label:
            return None
        chosen = {r["id"]: r["courses"][0] for r in plan["requirements"] if r["category"] == "ge" and r["courses"]}
        group_of = {s.spec.slot_id: s for s in built.variant}

        def item_ref(it: GEItem) -> dict:
            return {"type": it.kind, "ref": it.ref, "label": it.label}

        rows, lab_row = [], None
        for st in self.ge_states:
            spec = st.spec
            row = {"slot_id": spec.slot_id, "requirement_id": spec.requirement_id, "area_id": spec.area_id,
                   "name": spec.name, "area_name": spec.area_name, "rule": spec.rule}
            if self.golden_four:
                row["golden_four_name"] = self.golden_four.area_name(spec.requirement_id) or spec.name
            if st.satisfied_by:
                row.update(status="completed" if st.satisfied_by.kind == "completed_course" else "ap",
                           satisfied_by=item_ref(st.satisfied_by))
            else:
                slot = group_of.get(spec.slot_id)
                pick = chosen.get(slot.group_id) if slot else None
                if pick is None:
                    row.update(status="not_planned")
                elif pick["code"] in built.placeholders:
                    row.update(status="planned", term=pick["term"], units=built.catalog[pick["code"]].units,
                               slot_code=pick["code"], lab_required=slot.lab_mode == "with_lab")
                else:
                    # A real course fills it: a major-prep course (overlap), or a GE course planned by name because
                    # another GE requirement's courses need it (e.g. ENGL C1000 for Cal-GETC 1B, see _ge_prereqs)
                    major = pick["kind"] == "major"
                    row.update(status="completed" if pick["status"] == "completed" else "major_prep" if major else "planned",
                               satisfied_by={"type": "major_prep" if major else "course",
                                             "ref": pick["code"], "label": f"{pick['code']} {pick['title']}"},
                               term=pick["term"], lab_required=slot.lab_mode == "with_lab")
            rows.append(row)
        extra = [s for s in built.variant if s.lab_mode == "lab_only"]
        for slot in extra:
            pick = chosen.get(slot.group_id)
            rows.append({"slot_id": slot.spec.slot_id, "requirement_id": slot.spec.requirement_id,
                         "area_id": slot.spec.area_id, "name": slot.spec.name, "area_name": slot.spec.area_name,
                         "rule": None, "status": "planned" if pick else "not_planned",
                         "term": pick["term"] if pick else None, "slot_code": pick["code"] if pick else None})
        if self.ge_ds and self.ge_ds.lab and self.ge_pid == CAL_GETC:
            with_lab = next((r for r in rows if r.get("lab_required") or r["slot_id"] == self.ge_ds.lab["subarea_id"]), None)
            if self.ge_lab_item:
                lab_row = {"required": True, "status": "completed" if self.ge_lab_item.kind == "completed_course" else "ap",
                           "satisfied_by": item_ref(self.ge_lab_item)}
            elif with_lab:
                lab_row = {"required": True, "status": with_lab["status"], "via_requirement": with_lab["requirement_id"],
                           "term": with_lab.get("term")}
            else:
                lab_row = {"required": True, "status": "not_planned"}
        ds, g4 = self.ge_ds, self.golden_four
        own = {c for c, m in built.placeholders.items() if not g4 or m.get("role") == "golden_four"}
        if g4 and ds:
            rules = [f"Golden Four: {ds.display_name} {', '.join(g4.area_ids)} from {ds.institution_name}'s list"
                     + (f", each with a grade of {g4.minimum_grade} or better" if g4.minimum_grade else "")
                     + ". A CSU admission minimum, not a GE certification."]
        else:
            rules = ["Each course counts toward one area; a 5C lab may also count with its 5A/5B area (as the "
                     "college list states)."] if ds else []
        return {
            "pathway": self.ge_pid, "display_name": self.ge_label, "institution_id": self.college_id,
            "dataset_id": ds.dataset_id if ds else (self.data.ge.uc_seven.dataset_id if self.data.ge.uc_seven else None),
            "academic_year": ds.academic_year if ds else None, "requested_academic_year": self.wanted,
            "year_status": year_status(ds.academic_year, self.wanted) if ds else "unknown",
            "course_list": "loaded" if ds else "not_loaded",
            "course_list_name": ds.display_name if ds else None,
            "ge_certification": self.ge_pid == CAL_GETC,
            "requirements": rows, "lab": lab_row,
            "planned_units": sum(built.catalog[c].units for s in plan["semesters"] for c in
                                 (x["code"] for x in s["courses"]) if c in own),
            "rules": rules,
        }

    # =================================================================
    # CSU Golden Four: the CSU upper-division transfer minimums
    # =================================================================
    def _course_units(self, code: str) -> float | None:
        info = self._units_of(code_key(code))
        if info:
            return float(info[2])
        g = self.ge_ds.course(code) if self.ge_ds else None
        return float(g.units) if g and g.units else None

    def _csu_tally(self, built: Built, choices: dict[str, list[str]], planned) -> dict:
        """
        What counts toward the CSU transfer minimums for one course selection (choices: requirement group id ->
        the courses counted toward it; planned: every course still to take).

        GE units: each course counts once, toward one distinct area of the college's Cal-GETC list: the Golden
        Four first (as matched in _golden_four_state), then the plan's GE slots, then any other planned course on
        the list (e.g. PHYS 4A -> 5A). A second course in an area already counted isn't counted. AP exams fill an
        area only as the college's AP chart states, and add no units: no loaded data gives their CSU unit value.
        Transferable units: a course counts only when the data shows it transfers to the CSU (articulated in the
        loaded CSU agreement, or on the college's Cal-GETC list, GE slots included); an elective slot stands for a
        CSU-transferable course by definition. Anything else is listed as unverified, never assumed.
        """
        ds, g4, planned = self.ge_ds, self.golden_four, set(planned)
        done = set(self.completed_real) | set(self.ap_waived)
        rows, used = [], set()

        def add(spec: SlotSpec, code: str | None, label: str, units: float, status: str, via: str, golden: bool):
            rows.append({"requirement_id": spec.requirement_id, "slot_id": spec.slot_id, "area_id": spec.area_id,
                         "name": spec.name, "golden_four": golden, "code": code, "label": label,
                         "units": units, "status": status, "via": via})

        for golden, states in ((True, self.ge_states), (False, self.csu_other_states)):
            for st in states:
                it = st.satisfied_by
                if it is None:
                    continue
                if it.kind == "ap_exam":
                    add(st.spec, None, it.label, 0.0, "ap", "ap_exam", golden)
                else:
                    add(st.spec, it.ref, it.label, self._course_units(it.ref) or 0.0, "completed", "completed_course",
                        golden)
                    used.add(it.ref)
        for slot in built.variant:
            for code in choices.get(slot.group_id, []):
                ph = built.placeholders.get(code)
                add(slot.spec, None if ph else code, ph["label"] if ph else f"{code} {built.catalog[code].title}",
                    float(built.catalog[code].units), "completed" if code in done else "planned",
                    "ge_slot" if ph else "course", slot.role == "golden_four")
                used.add(code)
        filled = {r["slot_id"] for r in rows}
        items = [it for c in sorted(planned) if c not in used and c not in built.placeholders
                 and c not in built.electives for it in [course_item(ds, c)] if it]
        states, _ = match_items([sp for sp in self.csu_other_specs if sp.slot_id not in filled], False, items)
        for st in states:
            if st.satisfied_by:
                code = st.satisfied_by.ref
                add(st.spec, code, f"{code} {built.catalog[code].title}", float(built.catalog[code].units), "planned",
                    "listed_course", False)

        articulated, listed = set(self._course_index()), set(ds.courses)

        def basis(code: str) -> str | None:
            if code in built.electives:
                return "elective"
            if code in built.placeholders:
                return "ge_list"
            k = code_key(code)
            return "articulated" if k in articulated else "ge_list" if k in listed else None

        t_rows = [{"code": c, "units": u, "status": "completed", "basis": basis(c)}
                  for c in self.completed_real if (u := self._course_units(c))]
        t_rows += [{"code": c, "units": float(built.catalog[c].units), "status": "planned", "basis": basis(c)}
                   for c in sorted(planned)]
        counted = [r for r in t_rows if r["basis"]]

        def total(rs: list[dict], status: str | None = None) -> float:
            return sum(r["units"] for r in rs if status is None or r["status"] == status)

        return {
            "ge_units": {"required": g4.ge_units, "completed": total(rows, "completed"),
                         "planned": total(rows, "planned"), "total": total(rows), "courses": rows,
                         "covered_slots": {r["slot_id"] for r in rows}},
            "transferable_units": {"required": g4.transferable_units, "completed": total(counted, "completed"),
                                   "planned": total(counted, "planned"), "total": total(counted), "courses": counted,
                                   "elective_units": total([r for r in counted if r["basis"] == "elective"]),
                                   "unverified": [r for r in t_rows if not r["basis"]]},
        }

    def _transfer_requirements(self, plan: dict, built: Built, tally: dict) -> dict:
        """plan["transfer_requirements"] (CSU Golden Four mode): each CSU upper-division minimum and where the plan
        stands on it. What the planner can't know (grades, GPA, standing, campus criteria) is marked as such."""
        g4, p = self.golden_four, self.pathway
        order = {s["term"]: s["index"] for s in plan["semesters"]}
        areas = [{"requirement_id": r["requirement_id"], "name": r.get("golden_four_name") or r["name"],
                  "cal_getc_name": r["name"], "status": r["status"], "satisfied_by": r.get("satisfied_by"),
                  "term": r.get("term")} for r in (plan.get("ge") or {}).get("requirements", [])]
        now = bool(areas) and all(a["status"] in ("completed", "ap") for a in areas)
        by_transfer = (len(areas) == len(g4.area_ids)
                       and all(a["status"] in ("completed", "ap", "major_prep", "planned") for a in areas))
        last = max((a["term"] for a in areas if a["term"] in order), key=order.get, default=None)
        gu, tu = tally["ge_units"], tally["transferable_units"]
        major = [c for r in plan["requirements"] if r["category"] == "major" for c in r["courses"]]
        n = _num
        ap_areas = [r["requirement_id"] for r in gu["courses"] if r["status"] == "ap"]
        ge_met = gu["total"] >= gu["required"] - 1e-9
        tu_met = tu["total"] >= tu["required"] - 1e-9
        # Short only because AP adds no units here (the campus awards them; no loaded data says how many): review
        return {
            "pathway": CSU_GOLDEN_FOUR, "type": "csu_upper_division_transfer_minimums", "dataset_id": g4.dataset_id,
            "source": {"document_title": g4.source.get("document_title"), "url": g4.source.get("url")},
            "cal_getc_certification": False,
            "golden_four": {"status": "complete" if now else "planned" if by_transfer else "incomplete",
                            "complete_before_transfer": by_transfer, "completion_term": last,
                            "minimum_grade": g4.minimum_grade, "areas": areas},
            "ge_units": {"required": n(gu["required"]), "completed": n(gu["completed"]), "planned": n(gu["planned"]),
                         "total": n(gu["total"]), "met": ge_met,
                         "status": "met" if ge_met else "needs_review" if ap_areas else "short",
                         "includes_golden_four": True, "ap_areas": ap_areas,
                         "courses": [{**r, "units": n(r["units"])} for r in gu["courses"]],
                         "rule": (f"At least {gu['required']:g} semester units of GE-level courses, including the "
                                  f"Golden Four. Counted from {self.ge_ds.institution_name}'s {self.ge_ds.display_name} "
                                  f"list, each course once toward a distinct area; AP adds no units here.")},
            "transferable_units": {"required": n(tu["required"]), "completed": n(tu["completed"]),
                                   "planned": n(tu["planned"]), "total": n(tu["total"]),
                                   "met": tu_met, "status": "met" if tu_met else "short",
                                   "elective_units": n(tu["elective_units"]),
                                   "unverified": [{**r, "units": n(r["units"])} for r in tu["unverified"]],
                                   "ap_units_counted": False,
                                   "rule": ("Counted only for courses the loaded data shows transfer to the CSU (the "
                                            "articulation agreement or the Cal-GETC list); electives stand for any "
                                            "CSU-transferable course.")},
            "major_preparation": {"status": "complete" if major and all(c["status"] == "completed" for c in major)
                                  else "planned",
                                  "remaining_after_transfer": [r["label"] for r in self._remaining(built.alt)]},
            "gpa": {"status": "not_evaluated", "minimum": g4.minimum_gpa,
                    "minimum_nonresident": g4.minimum_gpa_nonresident,
                    "message": "The planner doesn't collect grades, so GPA isn't checked."},
            "good_standing": {"status": "not_evaluated", "required": g4.good_standing_required},
            "campus_requirements": {
                "status": "needs_review",
                "message": (f"{p.university} and the {p.major} major may add criteria (impaction, GPA, deadlines for "
                            f"the Golden Four). No campus- or major-specific exception is in the loaded data."),
                "policies": [x.get("title") for x in (self.ag.policies if self.ag else []) if x.get("title")],
                "recommendations": [x.get("action") for x in (self.ag.recommendations if self.ag else [])
                                    if x.get("action")]},
        }

    def _golden_four_notes(self, plans: list[dict]) -> None:
        g4, p = self.golden_four, self.pathway
        if not g4 or not self.ge_ds:
            return
        self.note("transfer_pathway", (
            f"CSU Golden Four route: the plan targets the CSU minimums for upper-division transfer (the Golden Four, "
            f"{g4.ge_units:g} GE units, {g4.transferable_units:g} transferable units). It isn't Cal-GETC "
            f"certification, so remaining lower- and upper-division GE is completed after transfer, and campus or "
            f"major criteria may still apply."))
        tr = next((pl["transfer_requirements"] for pl in plans if pl.get("transfer_requirements")), None)
        if tr is None:
            return
        tu, gu = tr["transferable_units"], tr["ge_units"]
        if tu["unverified"]:
            self.note("transfer_units", (
                f"Not counted toward the {tu['required']} transferable units, since the loaded data doesn't show they "
                f"transfer to the CSU: " + ", ".join(f"{r['code']} ({r['units']} units)" for r in tu["unverified"])
                + ". If ASSIST lists them as CSU-transferable, fewer elective units are needed."))
        if tu["elective_units"]:
            self.note("transfer_units", (
                f"{tu['elective_units']} units of CSU-transferable electives reach {tu['required']} units. Any "
                f"CSU-transferable course works; none is named because the loaded data can't tell which ones are."))
        if gu["status"] == "needs_review":
            self.note("ge", f"GE-level units reach {gu['total']} of {gu['required']} without AP: AP covers "
                            f"{', '.join(gu['ap_areas'])}, and the CSU campus awards AP units (no loaded data gives "
                            f"them), so the {gu['required']}-unit total needs review with a counselor.")
        elif gu["status"] == "short":
            self.note("ge", f"GE-level units reach {gu['total']} of {gu['required']}: no other area of "
                            f"{self.ge_ds.institution_name}'s list could be added; check with a counselor.")
        if any((e.get("ge_effect") or {}).get("status") == "eligible" for e in self.ap_evals):
            self.note("ap", "AP exams fill GE areas as the college's AP chart states, but add no units to the CSU "
                            "GE-unit or transferable-unit totals: the campus decides AP units.")
        grade = f"a {g4.minimum_grade} or better in the Golden Four and GE courses; " if g4.minimum_grade else ""
        self.note("transfer_pathway", (
            f"CSU admission also needs {grade}GPA ({g4.minimum_gpa:.1f}+, {g4.minimum_gpa_nonresident:.1f}+ for "
            f"non-residents) and good standing. These aren't checked: the planner doesn't collect grades."))
        for rec in tr["campus_requirements"]["recommendations"]:
            self.note("articulation", f"{p.university}'s agreement recommends: {rec}")

    def _articulation_info(self, built: Built, plan: dict | None) -> dict:
        p, ag = self.pathway, self.ag
        info = {"source": p.source, "dataset_id": p.dataset_id, "file": p.file, "publisher": p.publisher,
                "url": p.source_url, "retrieved_on": p.retrieved_on if not ag else None,
                "academic_year": p.academic_year, "requested_academic_year": self.match.academic_year,
                "year_status": year_status(p.academic_year, self.match.academic_year),
                "degree": p.degree, "not_articulated": [r["label"] for r in self._remaining(built.alt)],
                "prerequisites_enforced": bool(self.prereg) or p.prerequisites_enforced}
        if not ag:
            return info
        chosen = {}
        if plan:
            for r in plan["requirements"]:
                if r["category"] == "major" and r["id"] != "major-prep":
                    chosen[r["id"]] = [c for c in r["courses"]]
        done = set(built.completed)
        required = {code_key(c) for c in built.alt.required}
        rows = []
        for r in ag.requirements.values():
            row = {**r.to_dict(), "source_expression": ag.expr_text(r.expr)}
            if built.alt.alt and r.id in built.alt.alt.remaining:
                row.update(status="remaining_after_transfer", chosen=[])
            elif r.id in chosen:
                picks = chosen[r.id]
                row.update(status="completed" if all(c["status"] == "completed" for c in picks) else "planned",
                           chosen=[c["code"] for c in picks])
            elif r.expr and (sets := [s for s in _dnf(r.expr) if s <= required]):
                codes = [ag.code(k) for k in sorted(sets[0])]
                row.update(status="completed" if all(c in done for c in codes) else "planned", chosen=codes)
            else:
                row.update(status="not_selected", chosen=[])
            rows.append(row)
        info.update(document_title=ag.source.get("document_title"), date_published=ag.source.get("date_published"),
                    schema_version=ag.schema_version, concentration=ag.concentration, requirement_scope=ag.scope,
                    complete_degree_requirements_in_source=ag.complete_degree, requirements=rows,
                    alternatives_total=len(ag.alternatives), chosen_path=built.alt.path,
                    transfer_policies=ag.policies, recommendations=ag.recommendations)
        if len(ag.alternatives) > 1:
            self.note("articulation", (
                f"This agreement can be met {len(ag.alternatives)} ways; the plan uses the one with the fewest units"
                + (f" ({'; '.join(built.alt.path)})" if built.alt.path else "") + ". The other ways are equally valid."))
        return info

    def _remaining(self, alt: AltSpec) -> list[dict]:
        if not self.ag:
            return [{"type": "remaining_after_transfer", "requirement_id": None, "label": label,
                     "reason": "no_course_articulated", "target_courses": []} for label in alt.remaining]
        out = []
        for rid in alt.remaining:
            r = self.ag.requirements[rid]
            out.append({"type": "remaining_after_transfer", "requirement_id": rid, "label": r.label,
                        "target_courses": [t.__dict__ for t in r.targets],
                        "reason": "no_course_articulated" if r.status == "no_course_articulated" else r.status,
                        "source_note": r.note})
        return out

    def _prereq_notes(self, resolved: ResolvedPrereqs, in_plan: set[str]) -> None:
        """One note per planned course whose prerequisite needs review (all details stay in review_items)."""
        by_course: dict[str, list] = {}
        for r in resolved.reviews:
            if r.course in in_plan:
                by_course.setdefault(r.course, []).append(r)
        for course, items in by_course.items():
            kinds = {r.kind for r in items}
            routes = list(dict.fromkeys(r.detail for r in items if r.kind == "condition_route" and r.detail))
            if "policy" in kinds:
                rec = self._record(code_key(course))
                msg = f"{course}: {rec.summary if rec and rec.summary else 'depends on placement or college policy'}"
                if routes or "condition" in kinds:
                    msg += " No prerequisite course was added; confirm placement with a counselor."
            elif routes:
                msg = (f"{course} prerequisite: {' and '.join(routes)}. This depends on placement or approval, so no "
                       f"prerequisite course was added; confirm with a counselor.")
            else:
                msg = " ".join(dict.fromkeys(r.message for r in items if r.kind not in (
                    "clarification", "same_term", "condition_route", "policy_satisfied")))
            if "clarification" in kinds:
                msg = (msg or f"{course}:") + (" (Prerequisite grouping from a user-supplied clarification, not "
                                               "independently verified.)")
            extra = [r.message for r in items if r.kind in ("same_term", "policy_satisfied")]
            text = " ".join(x for x in [msg.strip(), *dict.fromkeys(extra)] if x)
            if text:
                self.note("prerequisite", text, course)
        unconfirmed = sorted(set(resolved.unconfirmed_year) & in_plan)
        if unconfirmed:
            self.note("prerequisite", (
                f"Prerequisites for {', '.join(unconfirmed)} come from cited college sources but aren't confirmed "
                f"against the current catalog year."))
        added = sorted(c for c in resolved.added if c in in_plan)
        if added:
            self.note("prerequisite", "Added because a planned course lists it as a prerequisite: "
                      + ", ".join(f"{c} (for {resolved.added[c]})" for c in added) + ". Already met some other "
                      "way (placement or other coursework)? Confirm with a counselor: the loaded prerequisite data "
                      "names only these courses.")

    def _ge_order_notes(self, built: Built, in_plan: set[str]) -> None:
        """Why a GE slot sits after a course: the courses that can fill it list that course as a prerequisite."""
        for slot_code, dep in built.ge_links:
            meta = built.placeholders.get(slot_code)
            if meta is None or slot_code not in in_plan or dep not in in_plan:
                continue
            slot = next(s for s in built.variant if self._slot_code(s) == slot_code)
            names = sorted(self.ge_ds.courses[k].code if k in self.ge_ds.courses else split_code(k)
                           for k in self._slot_eligible(slot))
            self.note("ge", f"{meta['label']} is planned after {dep}, a prerequisite of the courses that can fill it "
                            f"({', '.join(names)}).", slot_code)

    def _source_notes(self) -> None:
        if self.catalog and year_status(self.catalog.academic_year, self.wanted) == "older_year":
            self.note("data_year", f"Course titles and units come from the {self.catalog.title}; a {self.wanted} "
                                   f"catalog isn't loaded.")
        if self.ge_ds and year_status(self.ge_ds.academic_year, self.wanted) == "older_year":
            self.note("data_year", f"GE courses come from {self.ge_ds.institution_name}'s {self.ge_ds.academic_year} "
                                   f"{self.ge_ds.display_name} list; the {self.wanted} list isn't loaded, so check "
                                   f"ASSIST for changes.")

    def _data_sources(self) -> dict:
        p, cat, ds = self.pathway, self.catalog, self.ge_ds
        college_ap = self.data.ap.college.get(self.college_id)
        campus_ap = self.data.ap.campuses.get(self.university_id)
        return {
            "articulation": {"status": "loaded", "file": p.file, "dataset_id": p.dataset_id,
                             "academic_year": p.academic_year, "requested_academic_year": self.match.academic_year,
                             "year_status": year_status(p.academic_year, self.match.academic_year)},
            "catalog": ({"status": "loaded", "file": cat.file, "academic_year": cat.academic_year,
                         "year_status": year_status(cat.academic_year, self.wanted)} if cat else {"status": "not_loaded"}),
            "prerequisites": ({"status": "loaded", "file": self.prereg.file, "reviewed_on": self.prereg.reviewed_on}
                              if self.prereg else {"status": "not_loaded"}),
            "ge": ({"status": "loaded", "pathway": self.ge_pid, "file": ds.file, "dataset_id": ds.dataset_id,
                    "academic_year": ds.academic_year, "year_status": year_status(ds.academic_year, self.wanted)}
                   if ds else {"status": "loaded" if self.ge_pid == UC_SEVEN and self.data.ge.uc_seven else "not_loaded",
                               "pathway": self.ge_pid}),
            **({"golden_four": {"status": "loaded", "file": self.golden_four.file,
                                "dataset_id": self.golden_four.dataset_id}} if self.golden_four else {}),
            "ap": {"college_chart": ({"status": "loaded", "file": college_ap.file, "academic_year": college_ap.academic_year}
                                     if college_ap else {"status": "not_loaded"}),
                   "target_campus_chart": ({"status": "loaded", "campus": campus_ap.name,
                                            "source_period": campus_ap.source_period}
                                           if campus_ap else {"status": "not_loaded"})},
        }

    def _admission_requirements(self) -> dict | None:
        """UC 7-course pattern: UC targets only (a UC admission requirement, not a GE certification)."""
        if self.system != "UC":
            return None
        uc = self.data.ge.uc_seven
        if uc is None:
            return {"uc_seven_course": {"status": "not_loaded"}}
        specs = []
        for r in uc.requirements:
            n = int(r.get("minimum_courses") or 1)
            specs += [SlotSpec(f"{r['id']}#{i + 1}", r["id"], r["id"], r["name"], r["name"], frozenset(), r.get("rule"),
                               r.get("minimum_semester_units_each"), False, int(r.get("minimum_distinct_areas") or 0))
                      for i in range(n)]
        ap_items = [GEItem("ap_exam", e["exam_id"], e["exam_name"], (uc.ap_exams.get(e["exam_id"]) or ["?"])[0],
                           tuple((r["id"], False) for a in uc.ap_exams.get(e["exam_id"], [])
                                 for r in uc.requirements if a in (r.get("accepts") or [])))
                    for e in self.ap_evals if e.get("exam_id") and e["score"] >= uc.ap_min_score]
        states, _ = match_items(specs, False, ap_items)
        rows = []
        for r in uc.requirements:
            mine = [s for s in states if s.spec.requirement_id == r["id"]]
            by_ap = [s.satisfied_by.label for s in mine if s.satisfied_by]
            rows.append({"id": r["id"], "name": r["name"], "minimum_courses": len(mine), "rule": r.get("rule"),
                         "satisfied_by_ap": by_ap, "remaining_courses": len(mine) - len(by_ap),
                         "status": "met_by_ap" if len(by_ap) == len(mine) else "unknown_course_list"})
        return {"uc_seven_course": {
            "status": "evaluated", "type": "uc_transfer_admission_requirement", "applies_to": "UC targets",
            "dataset_id": uc.dataset_id, "requirements": rows, "course_list": "not_loaded",
            "general_rules": uc.general_rules,
            "message": ("A UC admission requirement (not GE). AP exams are counted from the loaded chart; which "
                        f"{self.pathway.college} courses count isn't in the loaded data, so check ASSIST.")}}


def plan_request(req, match, data: DataRegistry) -> dict:
    return Planner(req, match, data).run()


def ge_options(data: DataRegistry, institution: str, pathway: str, requirement_id: str, academic_year: str | None,
               lab: bool = False) -> dict:
    """GE detail panel: the courses that can fill one requirement, from the college's GE list + catalog."""
    pid = pathway_id(pathway) or pathway
    inst = data.resolve(institution)
    if pid == UC_SEVEN:
        return {"status": "not_loaded", "pathway": pid, "requirement_id": requirement_id, "options": [],
                "message": "No college course list for the UC 7-course pattern is loaded; ASSIST marks eligible "
                           "courses UC-E, UC-M, UC-H, UC-B and UC-S."}
    ds = data.ge.college_list(inst, COURSE_LIST_OF.get(pid, pid), academic_year)
    if ds is None:
        return {"status": "not_loaded", "pathway": pid, "requirement_id": requirement_id, "options": [],
                "message": "No GE course list for this college is loaded for that academic year."}
    found = ds.subarea(requirement_id)
    lab_info = ds.lab
    if found is None:
        area = next((a for a in ds.areas if str(a.get("id")) == requirement_id), None)
        if area is None:
            return {"status": "unknown_requirement", "pathway": pid, "requirement_id": requirement_id, "options": [],
                    "message": f"{requirement_id} isn't a requirement in {ds.display_name}."}
        found = (area, {"id": requirement_id, "name": area.get("name"),
                        "eligible_course_codes": [c for s in area.get("subareas") or [] for c in s.get("eligible_course_codes") or []]})
    area, sub = found
    keys = [code_key(c) for c in sub.get("eligible_course_codes") or []]
    if lab and lab_info and sub.get("id") != lab_info["subarea_id"]:
        keys = [k for k in keys if k in lab_info["eligible"]]
    catalog = data.catalogs.for_institution(inst, academic_year)
    options = []
    for k in dict.fromkeys(keys):
        g = ds.courses.get(k)
        c = catalog.get(k) if catalog else None
        code = (g.code if g else c.code if c else split_code(k))
        other = [a for a in (g.areas if g else ()) if a != sub.get("id")]
        options.append({"code": code, "title": (g.title if g else c.title if c else ""),
                        "units": g.units if g and g.units else (c.units if c else None),
                        "also_counts_for": other, "same_as": list(g.same_as) if g else [],
                        "prerequisite": (g.prerequisite if g and g.prerequisite else (c.prerequisite_text if c else None)),
                        "in_catalog": c is not None,
                        "includes_lab": bool(lab_info and k in lab_info["eligible"])})
    options.sort(key=lambda o: (o["code"].split()[0], _code_num(o["code"])))
    return {"status": "loaded", "pathway": pid, "display_name": ds.display_name, "institution_id": inst,
            "institution": ds.institution_name, "requirement_id": sub.get("id"), "name": sub.get("name"),
            "area_id": str(area.get("id")), "area_name": area.get("name"), "rule": area.get("rule"),
            "lab_required": bool(lab), "note": sub.get("note"),
            "minimum_disciplines": area.get("minimum_disciplines"),
            "dataset_id": ds.dataset_id, "academic_year": ds.academic_year, "requested_academic_year": academic_year,
            "year_status": year_status(ds.academic_year, academic_year), "options": options}


def _elective_units(short: float) -> list[float]:
    """Elective slots for `short` missing transferable units: whole units, ELECTIVE_UNITS-sized courses (a few a
    unit bigger) that add up exactly, e.g. 10 -> [4, 3, 3]; at least one ELECTIVE_UNITS course."""
    total = max(int(ELECTIVE_UNITS), math.ceil(short - 1e-9))
    n = max(1, total // int(ELECTIVE_UNITS))
    return [float(total // n + (1 if i < total % n else 0)) for i in range(n)]


def _num(x: float) -> float | int:
    return int(x) if float(x).is_integer() else round(x, 2)


def _code_num(code: str) -> tuple:
    import re
    m = re.search(r"(\d+)", code)
    return (int(m[1]) if m else 0, code)


__all__ = ["PlanError", "ge_options", "node_id", "plan_request"]
