"""
Algorithm 3 — Semester scheduling as a Constraint Satisfaction Problem.

Calendar: terms run Spring -> (Summer) -> Fall -> Spring ... from the student's
start term, with real labels ("Fall 2026", "Summer 2027"). Summer only appears
when the student opts in.

Variables : each course still to take.
Domains   : term indices 0..H-1 in which the course is offered and fits the
            term's unit cap, narrowed by its prerequisite chain (it can't
            start before `depth` terms, and must leave room for its longest
            dependent chain).
Constraints:
  * prerequisite  — every in-plan prereq is in a strictly earlier term
  * co-requisite  — every in-plan coreq is in the same term or earlier
  * unit caps     — Fall/Spring <= max_units; Summer <= max_summer_units (half-length term)
  * difficulty cap (optional) — sum of difficulty scores per term <= cap
  * time conflict — no two courses in one term have overlapping meetings
  * term availability — Fall-only / Spring-only / no-summer courses stay put
  * transfer timeline — the plan's LAST term is always a Spring (transfer for
    Fall admission). If the courses would finish in a Fall, the plan is
    padded to end with the following Spring.

Summer bias: GE courses try Summer terms first; major prep tries Fall/Spring
first and, for each horizon, is only allowed into Summer at all if no
schedule of that length exists without it (a strict pass runs first with
major prep removed from Summer domains, then a relaxed pass).

Search: depth-first backtracking. The next variable is the one with the
fewest remaining values (MRV), ties broken by topological order. When a
course doesn't fit, the search moves it to the next candidate term, and if
nothing works it backtracks and moves an earlier course instead. After every
placement, forward checking removes now-impossible terms from the domains of
unplaced courses and abandons the branch as soon as any domain empties or the
remaining units can't fit.

Horizon: Spring-ending horizons H are tried shortest first under the HARD caps,
so the first feasible one is the earliest transfer (proven unless a horizon hit
the node budget). A plan can also demand a minimum number of Fall/Spring terms
(e.g. a 3-year track) or a maximum average load (e.g. <= 15 units/term).

Load balancing (instead of packing early terms to the cap):
  Target_Units_Per_Term = total units / weighted term count, where a Summer
  counts as summer_cap/max_units of a term (9/18 = 0.5) and gets that share of
  the target. Within the chosen horizon the CSP is re-solved with a SOFT cap of
  target + 2 units per term (loosened to +3, +5 if needed; the hard cap always
  wins). In this mode major prep is placed first along its prerequisite chain,
  then GE courses (biggest first) go into the term furthest below its target.
  Three more starts are built semester by semester (list scheduling: each term
  takes the READY courses whose delay would hurt most; two variants also pace
  major prep to each term's fair share, so GE is placed while each term is
  built instead of filling whatever the major courses leave). Each
  start is hill-climbed (single-course moves and two-course swaps). If the best
  schedule still has GE crowding or gaps, increase the crowding/hole penalties and
  refine each start, allowing the hard caps if needed so the soft unit target
  cannot block better GE spread or sequence continuity.
  The lowest-cost schedule wins. The cost is soft preferences only; every hard
  constraint (prereqs, coreqs, caps, offerings, conflicts) is checked on every move:
    critical path  MAJOR_LATE_WEIGHT x (LATE_BASE + pull) x term index, pull = downstream priority
                   (courses it unlocks + chain still ahead of it) for a chain start, a quarter of that
                   for a course with in-plan prereqs (its gap below already ties it to them), plus
                   CHAIN_START_WEIGHT x chain ahead x term index for a chain start; a GE course
                   pays GE_LATE_WEIGHT x term index (an extra GE goes early, not into the last terms), plus
                   UNLOCK_LATE_WEIGHT x pull x term index when it unlocks planned courses (ENGL C1000 before
                   the GE slot whose courses need it), plus Course.early x term index (a course the data marks
                   as best taken early, e.g. a CSU Golden Four requirement)
    series         a course series = courses linked by prereq/coreq edges within one subject (from the
                   prerequisite graph: CSCI 14 -> 15 -> 20 and CSCI 21 after 14; MTH 1 -> 2 -> 3 ...):
                   SERIES_IDLE_WEIGHT per Fall/Spring term inside the series' span with none of its courses,
                   charged once per series. CSCI 14, 15, -, 20 pays; 14, 15, 20, 21 doesn't
    continuity     for each course, gap = offered Fall/Spring terms idle since its LATEST in-plan
                   prereq (Summer and terms the course isn't offered never count):
                   GAP_WEIGHT x gap + CHAIN_GAP_WEIGHT x chain weight x gap^2 + LONG_GAP_WEIGHT x (gap - 1)^2,
                   + CHAIN_PROGRESS_WEIGHT x gap if the course still unlocks planned courses; chain weight =
                   chain still ahead + chain behind beyond one step, so A -> B -> C -> D runs in consecutive
                   terms while a one-step branch may wait
    technical load MAJOR_BALANCE_WEIGHT x (major-prep units in a term - its share)^2, share = major-prep
                   units x term target / total Fall-Spring target: no all-STEM terms up front with the GE
                   saved for the end
    subject mix    per term, SUBJECT_PAIR_WEIGHT per pair of same-prefix major courses
                   + SUBJECT_CROWD_WEIGHT x (count - 2)^2, so three MTH courses spread out
                   when they can (the prefix is only a workload heuristic, never a requirement)
    GE spread      whole-plan GE variance: GE_SPREAD_WEIGHT x sum((GE in term - term's share)^2), with
                   the share = remaining standalone GE x term target / total target; plus GE_CROWD_WEIGHT
                   x (GE - 2)^2, GE_HOLE_WEIGHT per term without GE and MIX_WEIGHT per all-GE term.
                   2,1,1,2,2,2 scores far better than 1,1,1,2,2,3 or 2,0,0,2,3,3
    units          UNIT_WEIGHT x (load - target)^2 inside the +-2 healthy band, UNIT_BAND_WEIGHT x
                   (excess + excess^2) outside it: the last preference, a 13-unit term that keeps GE spread
                   and a series moving beats a 16-unit one that doesn't
  Priority, roughly: series continuity > critical path > technical load > GE spread > units.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

from .errors import SchedulingError
from .models import Course, StudentProfile, Term
from .toposort import chain_metrics

_EPS = 1e-9
DEFAULT_SUMMER_MAX_UNITS = 9

# Load-balancing objective (see the module docstring). These are SOFT preferences only; every hard
# constraint (prereqs, coreqs, caps, offerings, conflicts) is enforced by the search itself.
# Priority: series continuity and the critical sequence > technical load balance > GE spread over the whole
# plan > subject mix > units near target. A course that unlocks nothing may yield a term so GE can spread.
UNIT_WEIGHT = 0.5                 # gentle pull toward the target inside the healthy band ...
UNIT_BAND = 2.0                   # ... of +-2 units around it,
UNIT_BAND_WEIGHT = 4.0            # x (excess + excess^2) beyond the band; always a soft cost
                                  #   units come last: a 13-unit term is fine if it keeps GE spread and series moving
GE_SPREAD_WEIGHT = 8.0            # x sum over terms of (GE courses - that term's share)^2: whole-plan GE variance
GE_CROWD_WEIGHT = 20.0            # x (GE courses - GE_SOFT_MAX)^2: a third GE in one term only when unavoidable
GE_SOFT_MAX = 2
GE_HOLE_WEIGHT = 12.0             # per term without GE when there is enough GE to go around
GE_REPAIR_CROWD_WEIGHT = 80.0     # outweigh unit smoothing when construction caps leave GE crowding ...
GE_REPAIR_HOLE_WEIGHT = 24.0      # ... or terms without GE despite a full GE share
GE_LATE_WEIGHT = 3.0              # x term index per GE course: an extra GE goes early, not into the last terms
MIX_WEIGHT = 4.0                  # per term that is all GE
MAJOR_BALANCE_WEIGHT = 2.0        # x (major-prep units above that term's share + MAJOR_SLACK)^2: no STEM-heavy
MAJOR_SLACK = 2.0                 #   terms up front with the GE saved for the end (a light term is never pushed up)
SERIES_FACTOR = 1.5               # gap multiplier when the binding prereq shares the course's subject (a series) ...
SERIES_DEPTH = 3.0                # ... x (1 + this x chain still ahead): CS 15 (-> CS 20) is never delayed for GE
SERIES_IDLE_WEIGHT = 90.0         # per avoidable idle Fall/Spring term inside a course series (see _series_cost)
CHAIN_PROGRESS_WEIGHT = 15.0      # x gap for a course that still unlocks planned courses: a chain keeps moving
CHAIN_START_WEIGHT = 10.0         # x chain length ahead x term index for major prep that starts a chain (no in-plan
                                  #   prereq): a chain that could start now doesn't wait so GE can go first
MAJOR_LATE_WEIGHT = 4.0           # x (LATE_BASE + pull) x term index; pull = downstream priority for a chain
LATE_BASE = 0.5                   #   start, MID_CHAIN_LATE_FACTOR x that for a course with in-plan prereqs
MID_CHAIN_LATE_FACTOR = 0.25      #   (its gap penalty already ties it to its prereqs; slack lets it yield to GE)
UNLOCK_LATE_WEIGHT = 3.0          # x pull x term index for a non-major course that unlocks planned courses (a GE
                                  #   course another GE requirement's courses need); 0 for standalone GE
GAP_WEIGHT = 2.0                  # x gap, gap = idle Fall/Spring terms after the latest prereq ...
CHAIN_GAP_WEIGHT = 6.0            # + this x chain weight x gap^2 (chain still ahead + chain behind beyond one step)
LONG_GAP_WEIGHT = 1.0             # + this x (gap - 1)^2: waiting 2+ terms costs more even off a long chain
SUBJECT_PAIR_WEIGHT = 1.5         # per pair of same-subject major courses in one term
SUBJECT_CROWD_WEIGHT = 16.0       # x (same-subject courses - SUBJECT_SOFT_MAX)^2
SUBJECT_SOFT_MAX = 2


def subject_of(code: str) -> str:
    """Subject prefix used only as a workload-diversity heuristic: 'MTH 3' -> 'MTH', 'PHYS 7A' -> 'PHYS'."""
    m = re.match(r"\s*([A-Za-z&]+)", code)
    return m.group(1).upper() if m else code


@dataclass(frozen=True)
class TermSlot:
    index: int
    term: Term
    year: int

    @property
    def label(self) -> str:
        return f"{self.term.value} {self.year}"

    @property
    def is_summer(self) -> bool:
        return self.term is Term.SUMMER


def parse_term_label(label: str) -> tuple[Term, int]:
    """'Fall 2026' -> (Term.FALL, 2026)."""
    m = re.fullmatch(r"\s*(fall|spring|summer)\s+(\d{4})\s*", label, re.IGNORECASE)
    if not m:
        raise ValueError(f"term must look like 'Fall 2026', got {label!r}")
    return Term(m.group(1).capitalize()), int(m.group(2))


def term_sequence(start_term: Term, start_year: int, count: int, include_summer: bool) -> list[TermSlot]:
    """The next `count` terms starting at start_term/start_year, in calendar order."""
    cycle = [Term.SPRING, Term.SUMMER, Term.FALL] if include_summer else [Term.SPRING, Term.FALL]
    if start_term not in cycle:
        raise SchedulingError("Starting in Summer requires include_summer=True.")
    i, year, out = cycle.index(start_term), start_year, []
    for n in range(count):
        out.append(TermSlot(n, cycle[i], year))
        i += 1
        if i == len(cycle):     # wrapped past Fall -> next calendar year
            i, year = 0, year + 1
    return out


@dataclass
class Schedule:
    slots: list[TermSlot]
    assignment: dict[str, int]           # course -> term index
    caps: list[float]                    # hard unit cap of each slot
    proven_minimal: bool                 # False if a shorter horizon ran out of search budget
    nodes_explored: int
    lower_bound: int                     # terms needed ignoring availability/conflicts/Spring-end
    major_in_summer: list[str] = field(default_factory=list)   # major prep that had to go in Summer
    targets: list[float] = field(default_factory=list)         # load-balancing target per slot (units)
    soft_caps: list[float] = field(default_factory=list)       # construction caps, widened for accepted improvements

    def terms(self) -> list[tuple[TermSlot, list[str]]]:
        return [(slot, [c for c, t in self.assignment.items() if t == slot.index]) for slot in self.slots]


class _BudgetExceeded(Exception):
    pass


def regular_term_count(slots: list[TermSlot]) -> int:
    return sum(not s.is_summer for s in slots)


def weighted_terms(slots: list[TermSlot], summer_weight: float) -> float:
    """Capacity-weighted term count: a Summer counts as `summer_weight` of a Fall/Spring (9/18 = 0.5)."""
    n_regular = regular_term_count(slots)
    return n_regular + summer_weight * (len(slots) - n_regular)


def spring_horizon(profile: StudentProfile, min_regular_terms: int) -> list[TermSlot]:
    """Shortest Spring-ending run of terms from the start term with at least `min_regular_terms` Fall/Spring terms."""
    for h in range(1, 40):
        slots = term_sequence(profile.start_term, profile.start_year, h, profile.include_summer)
        if slots[-1].term is Term.SPRING and regular_term_count(slots) >= min_regular_terms:
            return slots
    raise SchedulingError(f"No Spring-ending horizon with {min_regular_terms} regular terms.")


class _Backtracker:
    """
    DFS backtracking + forward checking over term indices.

    Two modes:
      * feasibility (targets=None): MRV variable order; GE Summer-first, major prep Fall/Spring-first,
        then earliest term. Decides the shortest valid horizon, exactly as before.
      * load-balanced (targets given): major prep is placed first (MRV among majors, earliest term within
        the soft caps, i.e. along the prerequisite chain), then GE courses are placed biggest-first into
        the term furthest below its target ("padding" the light terms).
    list_schedule() builds an alternative start term by term; rebalance() hill-climbs either start on the
    soft objective (_term_cost + _course_cost).
    """

    def __init__(self, catalog: Mapping[str, Course], order: list[str], slots: list[TermSlot],
                 caps: list[float], max_difficulty: float | None, major: set[str],
                 node_budget: int, targets: list[float] | None = None) -> None:
        in_plan = set(order)
        self.course = {c: catalog[c] for c in order}
        self.rank = {c: i for i, c in enumerate(order)}
        self.prereqs = {c: {p for p in catalog[c].prereqs if p in in_plan} for c in order}
        self.coreqs = {c: {q for q in catalog[c].coreqs if q in in_plan} for c in order}
        self.dependents: dict[str, set[str]] = {c: set() for c in order}
        self.coreq_of: dict[str, set[str]] = {c: set() for c in order}
        for c in order:
            for p in self.prereqs[c]:
                self.dependents[p].add(c)
            for q in self.coreqs[c]:
                self.coreq_of[q].add(c)

        self.slots = slots
        self.caps = caps
        self.max_difficulty = max_difficulty
        self.major = major
        self.targets = targets
        self.ge_crowd_weight = GE_CROWD_WEIGHT
        self.ge_hole_weight = GE_HOLE_WEIGHT
        self.load = [0.0] * len(slots)
        self.difficulty = [0.0] * len(slots)
        self.members: list[list[str]] = [[] for _ in slots]
        self.assignment: dict[str, int] = {}
        self.nodes = 0
        self.node_budget = node_budget

        # Soft-objective metadata. Subject is only a diversity heuristic, never a requirement.
        self.subject = {c: subject_of(c) for c in order}
        below: dict[str, int] = {}
        for c in reversed(order):                                 # longest in-plan chain after c (edges)
            below[c] = max((1 + below[d] for d in self.dependents[c]), default=0)
        above: dict[str, int] = {}
        for c in order:                                           # longest in-plan chain before c (edges)
            above[c] = max((1 + above[p] for p in self.prereqs[c]), default=0)
        # Downstream priority: courses it unlocks + length of the chain still ahead of it.
        self.priority = {c: len(self._descendants(c)) + below[c] for c in order}
        # Chain weight for gaps: the chain still AHEAD of c (what a delay pushes back) plus the chain behind it
        # beyond one step. 0 for a one-step branch (MATH 1 -> linear algebra), 1 for the tail of CS 14 -> 15 -> 20,
        # 3 for B in A -> B -> C -> D: pauses there stretch the plan out.
        self.chain_len = {c: below[c] + max(0, above[c] - 1) for c in order}
        self.below = below
        # open_before[c][i] = Fall/Spring terms before index i in which c is offered. A gap only counts terms the
        # course could really have been taken: Summer never counts, nor does a Spring for a Fall-only course.
        self.open_before = {}
        for c, course in self.course.items():
            counts = [0]
            for slot in slots:
                counts.append(counts[-1] + (not slot.is_summer and slot.term in course.terms_offered))
            self.open_before[c] = counts
        self.series = self._series_families()
        self.major_share = [0.0] * len(slots)
        if targets is not None:
            self._set_ge_targets(targets)

    def _series_families(self) -> dict[str, frozenset[str]]:
        """
        Course series from the prerequisite graph: courses linked by prereq/coreq edges within one subject
        (CSCI 14 -> 15 -> 20 with CSCI 21 after 14; MTH 1 -> 2 -> 3 ...). Only the graph decides membership;
        the subject prefix just keeps a cross-subject chain (MTH 3 -> PHYS 7C) out of a series. course -> its
        series (courses outside any series map to nothing).
        """
        parent = {c: c for c in self.course}

        def root(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for c in self.course:
            for p in (*self.prereqs[c], *self.coreqs[c]):
                if self.subject[p] == self.subject[c]:
                    parent[root(p)] = root(c)
        groups: dict[str, set[str]] = {}
        for c in self.course:
            groups.setdefault(root(c), set()).add(c)
        return {c: frozenset(groups[root(c)]) for c in self.course if len(groups[root(c)]) > 1}

    def _set_ge_targets(self, targets: list[float]) -> None:
        """Each term's share of the standalone GE courses and of the major-prep units, proportional to its unit
        target (major prep's share only over Fall/Spring: it stays out of Summer when it can)."""
        n_ge = sum(c not in self.major for c in self.course)
        total = sum(targets) or 1.0
        self.ge_targets = [n_ge * t / total for t in targets]
        regular = [0.0 if slot.is_summer else t for slot, t in zip(self.slots, targets)]
        major_units = sum(self.course[c].units for c in self.major if c in self.course)
        self.major_share = [major_units * t / (sum(regular) or 1.0) for t in regular]

    def initial_domains(self, depth: dict[str, int], height: dict[str, int],
                        major_out_of_summer: bool) -> dict[str, list[int]]:
        H = len(self.slots)
        domains = {}
        for c, course in self.course.items():
            banned_summer = major_out_of_summer and c in self.major
            domains[c] = [s for s in range(depth[c], H - height[c] + 1)
                          if self.slots[s].term in course.terms_offered
                          and course.units <= self.caps[s] + _EPS
                          and not (banned_summer and self.slots[s].is_summer)]
        return domains

    def solve(self, domains: dict[str, list[int]]) -> dict[str, int] | None:
        if any(not d for d in domains.values()):
            return None
        return dict(self.assignment) if self._search(domains) else None

    # --- constraint checks -------------------------------------------------
    def _fits(self, c: str, s: int) -> bool:
        """Unit cap, difficulty cap and time conflicts against what's already in term s."""
        course = self.course[c]
        if self.load[s] + course.units > self.caps[s] + _EPS:
            return False
        if self.max_difficulty is not None and self.difficulty[s] + course.difficulty > self.max_difficulty + _EPS:
            return False
        return not any(course.conflicts_with(self.course[o]) for o in self.members[s])

    def _consistent(self, c: str, s: int) -> bool:
        a = self.assignment
        return (self._fits(c, s)
                and all(a[p] < s for p in self.prereqs[c] if p in a)
                and all(a[d] > s for d in self.dependents[c] if d in a)
                and all(a[q] <= s for q in self.coreqs[c] if q in a)
                and all(a[d] >= s for d in self.coreq_of[c] if d in a))

    def _place(self, c: str, s: int) -> None:
        self.assignment[c] = s
        self.load[s] += self.course[c].units
        self.difficulty[s] += self.course[c].difficulty
        self.members[s].append(c)

    def _unplace(self, c: str, s: int) -> None:
        del self.assignment[c]
        self.load[s] -= self.course[c].units
        self.difficulty[s] -= self.course[c].difficulty
        self.members[s].remove(c)

    # --- search --------------------------------------------------------------
    def _value_order(self, c: str, domain: list[int]) -> list[int]:
        is_major = c in self.major
        if self.targets is None:
            # Feasibility: GE Summer-first, major prep Fall/Spring-first, then earliest.
            return sorted(domain, key=lambda s: (0 if self.slots[s].is_summer != is_major else 1, s))
        if is_major:
            # Major prep: earliest Fall/Spring term within the soft cap (follows the prereq chain). A course that
            # unlocks nothing treats a term already holding SUBJECT_SOFT_MAX of its subject as 1.5 terms later,
            # so parallel siblings (three math courses all unlocked by one prereq) spread out instead of piling up.
            def crowded(s: int) -> bool:
                return not self.dependents[c] and SUBJECT_SOFT_MAX <= sum(
                    self.subject[o] == self.subject[c] for o in self.members[s] if o in self.major)
            return sorted(domain, key=lambda s: (1 if self.slots[s].is_summer else 0,
                                                 s + (1.5 if crowded(s) else 0), s))
        # GE: an under-target Summer first (GE goes to Summer), then the term furthest below its target.
        t = self.targets
        return sorted(domain, key=lambda s: (0 if self.slots[s].is_summer and self.load[s] < t[s] else 1,
                                             self.load[s] - t[s], s))

    def _pick_variable(self, domains: dict[str, list[int]]) -> str:
        if self.targets is None:
            return min(domains, key=lambda x: (len(domains[x]), self.rank[x]))            # MRV
        # Major prep first (MRV), then GE biggest-first so the fine-grained courses fill the gaps last.
        return min(domains, key=lambda x: (x not in self.major, len(domains[x]),
                                           0 if x in self.major else -self.course[x].units, self.rank[x]))

    def list_schedule(self, domains: dict[str, list[int]], targets: list[float], pace: str = "") -> bool:
        """
        Constructive semester-by-semester alternative to the backtracking start (starts from an empty
        schedule). In each term, the READY courses are those whose in-plan prereqs are all in earlier terms
        (coreqs no later) and that the term can still hold. Among them it repeatedly takes the one whose delay
        would hurt most (continuing a sequence, critical path, what it unlocks) net of what it does to the
        term's mix (subject crowding, GE mix), until the term reaches its unit target. A course at the last term
        of its domain must go now. Pacing holds major prep back once the term has its fair share of major
        courses (remaining major prep / remaining Fall-Spring terms), so GE is placed while each term is
        built rather than as leftovers: pace="leaves" holds back only courses that unlock nothing,
        pace="all" every major course that isn't at its deadline (the highest-priority ones still go first).
        Returns False if some course can't be placed (the caller keeps another start instead). Every hard
        constraint is checked by _consistent().
        """
        a = self.assignment
        allowed = {c: set(d) for c, d in domains.items()}
        unplaced = set(domains)
        regular_left = [sum(not t.is_summer for t in self.slots[s:]) for s in range(len(self.slots))]
        for s in range(len(self.slots)):
            quota = -(-sum(x in self.major for x in unplaced) // max(regular_left[s], 1))   # ceil
            while True:
                ready = [c for c in unplaced if s in allowed[c]
                         and all(p in a for p in self.prereqs[c]) and all(q in a for q in self.coreqs[c])
                         and self._consistent(c, s)]
                forced = [c for c in ready if max(domains[c]) == s]
                if pace and sum(x in self.major for x in self.members[s]) >= quota:
                    ready = [c for c in ready if c not in self.major or c in forced
                             or (pace == "leaves" and self.dependents[c])]
                if not forced and (not ready or self.load[s] >= targets[s] - 0.5):
                    break

                def score(c: str) -> tuple[float, int]:
                    # Units only decide when the term is full, not which course goes first (else the
                    # biggest course always wins while the term is still empty).
                    added = self._mix_cost(s, [*self.members[s], c]) - self._mix_cost(s, self.members[s])
                    later = lambda x: s + 1 if x == c else a[x]     # noqa: E731
                    now = lambda x: s if x == c else a[x]           # noqa: E731
                    delay = self._course_cost(c, later) - self._course_cost(c, now)
                    if c in self.series:                            # would its series sit idle this term?
                        delay += self._series_cost(self.series[c], later) - self._series_cost(self.series[c], now)
                    return added - delay, self.rank[c]
                c = min(forced or ready, key=score)
                self._place(c, s)
                unplaced.discard(c)
            if any(max(domains[c]) <= s for c in unplaced):
                return False
        return not unplaced

    def rebalance(self, domains: dict[str, list[int]], targets: list[float], max_moves: int = 500) -> int:
        """
        Hill-climb on a complete schedule: repeatedly make the single-course move, two-course swap or block move
        (a series, or a course with what it unlocks, shifted one term) that most lowers the load-balancing
        objective (module docstring), keeping every constraint (caps,
        offerings, prereqs, coreqs, time conflicts, major prep out of Summer when required). Swaps are what
        let a GE course trade places with an equal-sized major course once unit loads are already even.
        Returns the number of moves made.
        """
        self._set_ge_targets(targets)
        moves = 0
        while moves < max_moves:
            best = self._best_move(domains, targets)
            if best is None:
                return moves
            moved = best[1]
            for c, s, _ in moved:
                self._unplace(c, s)
            for c, _, t in moved:
                self._place(c, t)
            moves += 1
        return moves

    def _descendants(self, c: str) -> set[str]:
        seen, stack = set(), [c]
        while stack:
            x = stack.pop()
            for d in (*self.dependents[x], *self.coreq_of[x]):
                if d not in seen and d != c:
                    seen.add(d)
                    stack.append(d)
        return seen

    def _term_cost(self, s: int, members: list[str], targets: list[float]) -> float:
        """Per-term soft cost: units in a healthy band around the target, plus the course mix."""
        d = sum(self.course[c].units for c in members) - targets[s]
        excess = max(0.0, abs(d) - UNIT_BAND)
        return UNIT_WEIGHT * d * d + UNIT_BAND_WEIGHT * (excess + excess * excess) + self._mix_cost(s, members)

    def _mix_cost(self, s: int, members: list[str]) -> float:
        """GE spread/crowding, major-GE mix, technical load and subject concentration of one term (units aside)."""
        n_ge = sum(c not in self.major for c in members)
        subjects = Counter(self.subject[c] for c in members if c in self.major)
        major_units = sum(self.course[c].units for c in members if c in self.major)
        return (GE_SPREAD_WEIGHT * (n_ge - self.ge_targets[s]) ** 2
                + self.ge_crowd_weight * max(0, n_ge - GE_SOFT_MAX) ** 2
                + self.ge_hole_weight * (n_ge == 0 and self.ge_targets[s] >= 1)   # no GE although GE could spread
                + MIX_WEIGHT * (bool(members) and n_ge == len(members))           # all-GE term
                + MAJOR_BALANCE_WEIGHT * max(0.0, major_units - self.major_share[s] - MAJOR_SLACK) ** 2
                + sum(SUBJECT_PAIR_WEIGHT * n * (n - 1) / 2
                      + SUBJECT_CROWD_WEIGHT * max(0, n - SUBJECT_SOFT_MAX) ** 2 for n in subjects.values()))

    def _course_cost(self, c: str, term_of: Callable[[str], int]) -> float:
        """
        Per-course soft cost under a (possibly hypothetical) placement:
          lateness — chain starts that unlock a lot sit early (critical path first); a course with in-plan
                     prereqs is mostly held by its gap instead, so one with slack can yield a term to GE; a chain
                     start also pays CHAIN_START_WEIGHT x chain ahead per term, so GE never delays a chain. A GE
                     course pays GE_LATE_WEIGHT per term, so an extra GE goes early rather than into the last terms;
          gap      — offered Fall/Spring terms left idle between the course and its LATEST in-plan prereq (the
                     term it became ready): linear, plus quadratic in the chain length through the course, plus
                     a little extra past one term. A long chain moves A, B, C, D in consecutive terms while a
                     one-step branch (MATH 1 -> linear algebra) may wait for GE balance. Weighted more when the
                     prereq is in the same subject, and when the course still unlocks planned courses. A course
                     series pausing as a whole is charged once per series (_series_cost).
        """
        s = term_of(c)
        pull = self.priority[c] * (MID_CHAIN_LATE_FACTOR if self.prereqs[c] else 1.0)
        # Major prep sits early along the critical path; GE pays a flat lateness, more when it unlocks planned courses
        cost = (MAJOR_LATE_WEIGHT * (LATE_BASE + pull) if c in self.major
                else UNLOCK_LATE_WEIGHT * pull + GE_LATE_WEIGHT) * s
        if c in self.major and not self.prereqs[c] and self.below[c]:
            cost += CHAIN_START_WEIGHT * self.below[c] * s
        cost += self.course[c].early * s                              # e.g. CSU Golden Four goes early
        if self.prereqs[c]:
            ready = max(term_of(p) for p in self.prereqs[c])
            gap = self.open_before[c][s] - self.open_before[c][ready + 1]
            if gap > 0:
                series = any(term_of(p) == ready and self.subject[p] == self.subject[c] for p in self.prereqs[c])
                mult = SERIES_FACTOR * (1 + SERIES_DEPTH * self.below[c]) if series else 1.0
                cost += (mult * (GAP_WEIGHT * gap + CHAIN_GAP_WEIGHT * self.chain_len[c] * gap * gap
                                 + LONG_GAP_WEIGHT * (gap - 1) ** 2)
                         + CHAIN_PROGRESS_WEIGHT * gap * (self.below[c] > 0))    # it still leads on: don't stall
        return cost

    def _series_cost(self, series: frozenset[str], term_of: Callable[[str], int]) -> float:
        """
        SERIES_IDLE_WEIGHT per Fall/Spring term between a series' first and last planned course in which none of
        its courses is taken although a later one could have been (offered then, every prereq in an earlier term,
        coreqs no later): CSCI 14, 15, (nothing), 20 pays for the empty term; CSCI 14, 15, 20, then CSCI 21
        (which needs only CSCI 14) pays nothing, the series never stopped; ENGR 1, (nothing), ENGR 2 pays nothing
        when ENGR 2 also waits for PHYS 2. Courses not placed yet (list scheduling) are skipped.
        """
        when: dict[str, int] = {}
        for c in series:
            try:
                when[c] = term_of(c)
            except KeyError:
                pass
        if len(set(when.values())) < 2:
            return 0.0

        def takeable(c: str, t: int) -> bool:
            try:
                return (self.slots[t].term in self.course[c].terms_offered
                        and all(term_of(p) < t for p in self.prereqs[c]) and all(term_of(q) <= t for q in self.coreqs[c]))
            except KeyError:
                return False
        used = set(when.values())
        return SERIES_IDLE_WEIGHT * sum(1 for t in range(min(used) + 1, max(used))
                                        if t not in used and not self.slots[t].is_summer
                                        and any(w > t and takeable(c, t) for c, w in when.items()))

    def objective(self, targets: list[float]) -> float:
        """Total soft cost of the current complete schedule (what rebalance() minimizes)."""
        a = self.assignment
        return (sum(self._term_cost(s, self.members[s], targets) for s in range(len(self.slots)))
                + sum(self._course_cost(c, a.__getitem__) for c in a)
                + sum(self._series_cost(f, a.__getitem__) for f in set(self.series.values())))

    def _delta(self, moved: list[tuple[str, int, int]], targets: list[float]) -> float:
        a = self.assignment
        new_term = {c: t for c, _, t in moved}
        after = {x: list(self.members[x]) for _, s, t in moved for x in (s, t)}
        for c, s, t in moved:
            after[s].remove(c)
            after[t].append(c)
        delta = sum(self._term_cost(x, after[x], targets) - self._term_cost(x, self.members[x], targets)
                    for x in after)
        affected = set(new_term) | {d for c in new_term for d in self.dependents[c]}   # gaps depend on prereqs
        moved_term = lambda c: new_term.get(c, a[c])   # noqa: E731
        delta += sum(self._course_cost(c, moved_term) - self._course_cost(c, a.__getitem__) for c in affected)
        return delta + sum(self._series_cost(f, moved_term) - self._series_cost(f, a.__getitem__)
                           for f in {self.series[c] for c in new_term if c in self.series})

    def _allowed(self, moved: list[tuple[str, int, int]]) -> bool:
        """Every constraint still holds after the move/swap (the schedule is restored either way)."""
        for c, s, _ in moved:
            self._unplace(c, s)
        placed, ok = [], True
        for c, _, t in moved:
            if not self._consistent(c, t):
                ok = False
                break
            self._place(c, t)
            placed.append((c, t))
        for c, t in placed:
            self._unplace(c, t)
        for c, s, _ in moved:
            self._place(c, s)
        return ok

    def _best_move(self, domains: dict[str, list[int]], targets: list[float]):
        best = None
        a = self.assignment
        courses = list(a)
        allowed = {c: set(domains[c]) for c in courses}
        candidates = [[(c, a[c], t)] for c in courses for t in domains[c] if t != a[c]]
        candidates += [[(c, a[c], a[d]), (d, a[d], a[c])] for i, c in enumerate(courses) for d in courses[i + 1:]
                       if a[c] != a[d] and a[d] in allowed[c] and a[c] in allowed[d]]
        # Block moves: a whole series, or a course with everything it unlocks, one term earlier or later. Single
        # moves can't slide a chain without first breaking it (which the series/continuity costs forbid)
        blocks = set(self.series.values()) | {frozenset({c, *self._descendants(c)}) for c in courses if self.dependents[c]}
        for block in blocks:
            for d in (-1, 1):
                if all(a[c] + d in allowed[c] for c in block):
                    candidates.append([(c, a[c], a[c] + d) for c in sorted(block, key=a.__getitem__, reverse=d > 0)])
        for moved in candidates:
            delta = self._delta(moved, targets)
            if delta > -_EPS or (best is not None and delta >= best[0]):
                continue
            if self._allowed(moved):
                best = (delta, moved)
        return best

    def _search(self, domains: dict[str, list[int]]) -> bool:
        if not domains:
            return True
        self.nodes += 1
        if self.nodes > self.node_budget:
            raise _BudgetExceeded
        c = self._pick_variable(domains)
        rest = {x: d for x, d in domains.items() if x != c}
        for s in self._value_order(c, domains[c]):
            if not self._consistent(c, s):
                continue
            self._place(c, s)
            pruned = self._forward_check(c, s, rest)
            if pruned is not None and self._search(pruned):
                return True
            self._unplace(c, s)                                           # backtrack
        return False

    def _forward_check(self, c: str, s: int, domains: dict[str, list[int]]) -> dict[str, list[int]] | None:
        """Prune values made impossible by placing c in term s; None means a dead end."""
        pre, dep, co, co_of = self.prereqs[c], self.dependents[c], self.coreqs[c], self.coreq_of[c]
        pruned: dict[str, list[int]] = {}
        for x, dom in domains.items():
            keep = [v for v in dom
                    if not (x in pre and v >= s)
                    and not (x in dep and v <= s)
                    and not (x in co and v > s)
                    and not (x in co_of and v < s)
                    and (v != s or self._fits(x, s))]
            if not keep:
                return None
            pruned[x] = keep
        remaining = sum(self.course[x].units for x in pruned)
        capacity = sum(cap - load for cap, load in zip(self.caps, self.load))
        return pruned if remaining <= capacity + _EPS else None


def schedule_courses(
    catalog: Mapping[str, Course],
    order: list[str],
    profile: StudentProfile,
    max_units: float = 18,
    max_term_difficulty: float | None = None,
    *,
    max_summer_units: float = DEFAULT_SUMMER_MAX_UNITS,
    major_courses: Iterable[str] = (),
    balance: bool = True,
    balance_margin: float = 2.0,
    min_regular_terms: int = 0,
    max_avg_load: float | None = None,
    end_in_spring: bool = True,
    node_budget: int = 50_000,
) -> Schedule:
    """
    Assign topologically ordered courses to real terms.

    Horizon: the shortest Spring-ending run of terms that is feasible under the HARD caps and also has
    at least `min_regular_terms` Fall/Spring terms and (if given) an average load of at most
    `max_avg_load` units per Fall/Spring term (a Summer counts as summer_cap/max_units of a term).

    Load balancing (balance=True), within that horizon:
      Target_Units_Per_Term = total units / weighted term count  (Summer target scaled the same way)
      1. re-solve with each term capped at min(hard cap, target + balance_margin) as a soft cap,
         loosening the margin (+1, +3) if that's infeasible, else keeping the hard-cap solution;
      2. also build list-scheduled starts, hill-climb each start with moves and swaps, and refine under the
         hard caps if the best schedule still has GE crowding or gaps; keep the lowest
         soft cost: continuity, critical path, subject and GE mix, units near the target (module docstring).
    The hard caps (e.g. 18 units, 9 in Summer) are never exceeded.
    """
    summer_cap = min(max_units, max_summer_units)
    cap_for = {Term.FALL: max_units, Term.SPRING: max_units, Term.SUMMER: summer_cap}
    cycle_terms = {Term.FALL, Term.SPRING} | ({Term.SUMMER} if profile.include_summer else set())
    major = set(major_courses) & set(order)

    for c in order:
        course = catalog[c]
        if course.units > max_units:
            raise SchedulingError(f"{c} is {course.units} units, above the {max_units}-unit term cap.")
        if max_term_difficulty is not None and course.difficulty > max_term_difficulty:
            raise SchedulingError(f"{c} alone exceeds the per-term difficulty cap of {max_term_difficulty}.")
        usable = [t for t in course.terms_offered & cycle_terms if course.units <= cap_for[t] + _EPS]
        if not usable:
            offered = ", ".join(sorted(t.value for t in course.terms_offered))
            raise SchedulingError(f"{c} is only offered in {offered}, and no such term in this plan can "
                                  f"hold its {course.units} units (Summer cap {summer_cap}, "
                                  f"summer {'on' if profile.include_summer else 'off'}).")

    if not order:
        return Schedule([], {}, [], True, 0, 0)

    def slots_for(h: int) -> list[TermSlot]:
        return term_sequence(profile.start_term, profile.start_year, h, profile.include_summer)

    depth, height = chain_metrics(order, catalog)
    total_units = sum(catalog[c].units for c in order)
    lower = max(height.values())
    while lower < profile.max_terms and sum(cap_for[s.term] for s in slots_for(lower)) < total_units - _EPS:
        lower += 1

    summer_weight = summer_cap / max_units

    def summer_room(strict: bool) -> float:
        """Units that could go into Summer at all (offered in Summer, fit its cap; major prep only if allowed)."""
        return sum(catalog[c].units for c in order
                   if Term.SUMMER in catalog[c].terms_offered and catalog[c].units <= summer_cap + _EPS
                   and not (strict and c in major))

    proven, explored = True, 0
    for horizon in range(lower, profile.max_terms + 1):
        slots = slots_for(horizon)
        if end_in_spring and slots[-1].term is not Term.SPRING:
            continue                         # transfers happen after a Spring: only Spring-ending plans
        if regular_term_count(slots) < min_regular_terms:
            continue
        weight = weighted_terms(slots, summer_weight if summer_room(strict=True) > 0 else 0.0)
        if max_avg_load is not None and total_units / weight > max_avg_load + _EPS:
            continue                         # would need more than the allowed average load per term
        caps = [cap_for[s.term] for s in slots]
        has_summer = any(s.is_summer for s in slots)
        # Strict pass keeps major prep out of Summer; relax only if that finds nothing.
        for strict in ((True, False) if has_summer and major else (False,)):
            solver = _Backtracker(catalog, order, slots, caps, max_term_difficulty, major, node_budget)
            try:
                result = solver.solve(solver.initial_domains(depth, height, major_out_of_summer=strict))
            except _BudgetExceeded:
                result, proven = None, False
            explored += solver.nodes
            if result is None:
                continue
            targets, soft_caps = [], list(caps)
            if balance:
                targets = load_targets(slots, total_units, summer_weight, summer_room(strict))
                solver, soft_caps, n = _balance(catalog, order, slots, caps, targets, max_term_difficulty, major,
                                                depth, height, strict, balance_margin, node_budget, solver)
                explored += n
            assignment = dict(solver.assignment)
            in_summer = sorted(c for c in major if slots[assignment[c]].is_summer)
            return Schedule(slots, assignment, caps, proven, explored, lower, in_summer, targets, soft_caps)

    raise SchedulingError(
        f"No valid schedule ending in a Spring term fits within {profile.max_terms} terms under a "
        f"{max_units}-unit cap" + (f" and a difficulty cap of {max_term_difficulty}" if max_term_difficulty else "")
        + (f" with at least {min_regular_terms} Fall/Spring terms" if min_regular_terms else "")
        + (f" at no more than {max_avg_load:g} units per term on average" if max_avg_load else "")
        + ". Try allowing more terms, enabling summer, or relaxing the caps.")


def load_targets(slots: list[TermSlot], total_units: float, summer_weight: float, summer_room: float) -> list[float]:
    """
    Target_Units_Per_Term = total / weighted term count; a Summer's target is summer_weight of that.
    Summers only share the load if some course can go there, and never more than those courses' units.
    """
    n_regular = regular_term_count(slots)
    n_summer = len(slots) - n_regular
    w = summer_weight if summer_room > 0 else 0.0
    per_term = total_units / (n_regular + w * n_summer)
    summer_target = per_term * w
    if n_summer and summer_target * n_summer > summer_room:
        summer_target = summer_room / n_summer
        per_term = (total_units - summer_room) / n_regular
    return [summer_target if s.is_summer else per_term for s in slots]


def _balance(catalog, order, slots, caps, targets, max_difficulty, major, depth, height, strict,
             margin, node_budget, fallback: _Backtracker) -> tuple[_Backtracker, list[float], int]:
    """
    Soft-cap re-solve (tightest margin first), then the load-balancing hill-climb. Starts built semester
    by semester (_Backtracker.list_schedule) are hill-climbed too. If the best still has GE crowding or
    gaps, emphasize those penalties, then allow the hard caps if needed: a target-based cap must not
    veto better GE spread or continuity.
    The schedule with the lower soft cost wins, since the hill-climb alone can stall in a local optimum.
    Returns (solver, soft caps, nodes).
    """
    explored = 0
    solver, soft_caps = fallback, list(caps)
    for m in (margin, margin + 1, margin + 3):
        trial_caps = [min(cap, target + m) for cap, target in zip(caps, targets)]
        trial = _Backtracker(catalog, order, slots, trial_caps, max_difficulty, major,
                             min(node_budget, 20_000), targets=targets)
        try:
            found = trial.solve(trial.initial_domains(depth, height, major_out_of_summer=strict))
        except _BudgetExceeded:
            found = None
        explored += trial.nodes
        if found is not None:
            solver, soft_caps = trial, trial_caps
            break
    solver.rebalance(solver.initial_domains(depth, height, major_out_of_summer=strict), targets)
    candidates = [solver]

    best = solver.objective(targets)
    for pace in ("", "leaves", "all"):
        greedy = _Backtracker(catalog, order, slots, soft_caps, max_difficulty, major, node_budget, targets=targets)
        domains = greedy.initial_domains(depth, height, major_out_of_summer=strict)
        if all(domains.values()) and greedy.list_schedule(domains, targets, pace=pace):
            greedy.rebalance(domains, targets)
            candidates.append(greedy)
            if (cost := greedy.objective(targets)) < best - _EPS:
                solver, best = greedy, cost

    def uneven_ge(candidate: _Backtracker) -> bool:
        counts = [sum(c not in major for c in codes) for codes in candidate.members]
        return any(n > GE_SOFT_MAX or (n == 0 and target >= 1) for n, target in zip(counts, candidate.ge_targets))

    # Preserve healthy workloads when GE already spreads. Otherwise emphasize crowding and holes
    # in the same global objective, first within the construction caps, then using actual capacity
    # if the best schedule still needs it. Academic constraints are enforced in both passes.
    if uneven_ge(solver):
        best = float("inf")
        for candidate in candidates:
            candidate.ge_crowd_weight = GE_REPAIR_CROWD_WEIGHT
            candidate.ge_hole_weight = GE_REPAIR_HOLE_WEIGHT
            candidate.rebalance(candidate.initial_domains(depth, height, major_out_of_summer=strict), targets)
            if (cost := candidate.objective(targets)) < best - _EPS:
                solver, best = candidate, cost
        if uneven_ge(solver):
            for candidate in candidates:
                candidate.caps = list(caps)
                candidate.rebalance(candidate.initial_domains(depth, height, major_out_of_summer=strict), targets)
                if (cost := candidate.objective(targets)) < best - _EPS:
                    solver, best = candidate, cost
    return solver, [max(cap, load) for cap, load in zip(soft_caps, solver.load)], explored
