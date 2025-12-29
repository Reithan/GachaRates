from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import csv
import numpy as np


# -------------------------
# Parsing / utility helpers
# -------------------------

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _s(s: str) -> str:
    return (s or "").strip()


def parse_opt_int(s: str) -> Optional[int]:
    s = _s(s)
    return None if s == "" else int(s)


def parse_opt_float(s: str) -> Optional[float]:
    s = _s(s)
    return None if s == "" else float(s)


def parse_bool(s: str) -> bool:
    s = _s(s).lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off", ""):
        return False
    raise ValueError(f"Invalid boolean: {s}")


def normalize_name(name: str) -> str:
    # forgiving canonicalization for alias matching
    name = (name or "").lower()
    out = []
    for ch in name:
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


# -------------------------
# Parameter data structures
# -------------------------

@dataclass(frozen=True)
class LevelParams:
    # Hit-rate model
    B: float                    # base hit probability per pull (0..1)
    P: Optional[int] = None     # soft pity threshold (fails)
    R: float = 0.0              # linear increment per fail after P
    H: Optional[int] = None     # hard pity: guarantee hit when i_fail == H-1

    # Within-hit targeting model
    C: float = 0.0              # "rate-up" chance given a hit (when not forced)
    G: int = 1                  # guarantee after G-1 non-rate-up hits (1 => none)
    W: float = 0.0              # probability an "other" hit is desired
    rateup_desired: bool = False  # whether the rate-up hit counts as desired

    def validate(self) -> None:
        if not (0.0 < self.B <= 1.0):
            raise ValueError("B must be in (0,1].")
        if not (0.0 <= self.C <= 1.0):
            raise ValueError("C must be in [0,1].")
        if not (0.0 <= self.R <= 1.0):
            raise ValueError("R must be in [0,1].")
        if not (0.0 <= self.W <= 1.0):
            raise ValueError("W must be in [0,1].")
        if self.P is not None and self.P < 0:
            raise ValueError("P must be >= 0.")
        if self.H is not None and self.H < 1:
            raise ValueError("H must be >= 1.")
        if self.G < 1:
            raise ValueError("G must be >= 1.")


@dataclass(frozen=True)
class GameTemplate:
    canonical: str
    aliases: List[str]
    l1: LevelParams
    l2: Optional[LevelParams] = None
    notes: str = ""


# -------------------------
# Probability primitives
# -------------------------

def implied_H(lp: LevelParams) -> Optional[int]:
    """
    If H is omitted but (P,R) creates a linear ramp that reaches 1.0, infer an effective H.
    Returns H such that p(hit | i_fail == H-1) == 1.0.
    """
    if lp.H is not None:
        return lp.H
    if lp.P is None or lp.R <= 0.0:
        return None

    needed_steps = int(np.ceil((1.0 - lp.B) / lp.R))
    i_fail = (lp.P - 1) + needed_steps
    return max(i_fail + 1, 1)


def p_hit(i_fail: int, lp: LevelParams, H_eff: Optional[int]) -> float:
    if H_eff is not None and i_fail >= H_eff - 1:
        return 1.0
    p = lp.B
    if lp.P is not None and lp.R > 0.0 and i_fail >= lp.P:
        p = lp.B + lp.R * (i_fail - lp.P + 1)
    return clamp01(p)


def p_desired_on_unforced_hit(lp: LevelParams) -> float:
    # rate-up desired portion
    pr_des = lp.C if lp.rateup_desired else 0.0
    # "other" desired portion
    po_des = (1.0 - lp.C) * lp.W
    return clamp01(pr_des + po_des)


# -------------------------
# Single-level EV to desired
# -------------------------

def expected_pulls_single(lp: LevelParams, start_fail: int = 0, start_nonrate: int = 0) -> float:
    """
    Markov state: (i_fail, j_nonrate)
      i_fail: consecutive non-hit pulls
      j_nonrate: consecutive non-rate-up hits (capped at G-1)
    Absorption: a desired outcome.
    """
    lp.validate()

    H_eff = implied_H(lp)
    if H_eff is None:
        # No finite state possible if B<1 and no ramp/hard pity; but we can still solve:
        # with memoryless hit process, EV exists if desired probability per pull > 0.
        # Here we keep the model uniform by requiring finite H for state-based solve.
        # So: reduce to geometric over pulls using per-pull desired probability derived from:
        # P(hit)=B, P(desired | hit)=q_unforced with guarantee disabled only if G==1.
        #
        # If G>1 and no finite H, we'd still have a finite j-state but infinite i-state;
        # keep CLI simple: require some finite mechanism OR accept approximation.
        # We'll implement exact for G==1, else require finite H.
        if lp.G != 1:
            raise ValueError("For G>1, provide H or a soft pity ramp (P,R) that reaches 1.")
        q = p_desired_on_unforced_hit(lp)
        p_desired_per_pull = lp.B * q
        return float("inf") if p_desired_per_pull <= 0 else 1.0 / p_desired_per_pull

    G = lp.G
    n_states = H_eff * G

    def idx(i: int, j: int) -> int:
        return i * G + j

    A = np.zeros((n_states, n_states), dtype=float)
    b = np.ones((n_states,), dtype=float)

    for i in range(H_eff):
        for j in range(G):
            row = idx(i, j)
            A[row, row] = 1.0

            ph = p_hit(i, lp, H_eff)
            pf = 1.0 - ph

            # miss -> i+1, j unchanged
            i2 = min(i + 1, H_eff - 1)
            A[row, idx(i2, j)] -= pf

            # hit -> i resets to 0, then rate-up / other with guarantee
            forced_rateup = (j == G - 1) and (G > 1)
            if forced_rateup:
                # rate-up guaranteed
                if lp.rateup_desired:
                    # absorption with prob ph
                    pass
                else:
                    # not desired -> (0, 0)
                    A[row, idx(0, 0)] -= ph
            else:
                # unforced: rate-up with C, other with (1-C)
                pr = ph * lp.C
                po = ph * (1.0 - lp.C)

                # rate-up branch
                if lp.rateup_desired:
                    pass  # absorption
                else:
                    A[row, idx(0, 0)] -= pr

                # other branch: desired with prob W, else advance non-rate counter
                po_not_des = po * (1.0 - lp.W)
                j2 = min(j + 1, G - 1) if G > 1 else 0
                A[row, idx(0, j2)] -= po_not_des

    try:
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return float("inf")
    start_fail = max(0, min(start_fail, H_eff - 1))
    start_nonrate = max(0, min(start_nonrate, G - 1))
    return float(x[idx(start_fail, start_nonrate)])


# ---------------------------------------
# Split-level EV: L1, L2, and any desired
# ---------------------------------------

def expected_pulls_split(
    l1: LevelParams,
    l2: LevelParams,
    start_fail_1: int = 0,
    start_nonrate_1: int = 0,
    start_fail_2: int = 0,
    start_nonrate_2: int = 0,
) -> Tuple[float, float, float]:
    """
    Exclusive outcome model with precedence:
      On each pull:
        - L1 hit occurs with probability p1(i1)
        - If L1 does not hit, L2 hit occurs with probability p2(i2)
        - Otherwise neither hits.

    Counters:
      - L1 fail counter increments iff L1 does not hit. Resets on L1 hit.
      - L2 fail counter increments iff L2 does not hit. Resets on L2 hit OR L1 hit (default).
      - Non-rate counters (j1/j2) update only when that level hits; reset on rate-up hit.

    Outputs:
      EV to desired L1,
      EV to desired L2,
      EV to any desired (either level).
    """
    l1.validate()
    l2.validate()

    H1 = implied_H(l1)
    H2 = implied_H(l2)
    if H1 is None and l1.G != 1:
        raise ValueError("For L1 with G>1, provide H or a soft pity ramp that reaches 1.")
    if H2 is None and l2.G != 1:
        raise ValueError("For L2 with G>1, provide H or a soft pity ramp that reaches 1.")
    if H1 is None:
        raise ValueError("Split-rate requires finite-state L1 (provide H or P,R that reaches 1).")
    if H2 is None:
        raise ValueError("Split-rate requires finite-state L2 (provide H or P,R that reaches 1).")

    G1, G2 = l1.G, l2.G
    n_states = H1 * G1 * H2 * G2

    def idx(i1: int, j1: int, i2: int, j2: int) -> int:
        return (((i1 * G1 + j1) * H2 + i2) * G2 + j2)

    # helper to build and solve for a chosen absorption condition
    def solve(absorb_l1: bool, absorb_l2: bool) -> float:
        A = np.zeros((n_states, n_states), dtype=float)
        b = np.ones((n_states,), dtype=float)

        for i1 in range(H1):
            for j1 in range(G1):
                for i2 in range(H2):
                    for j2 in range(G2):
                        row = idx(i1, j1, i2, j2)
                        A[row, row] = 1.0

                        p1 = p_hit(i1, l1, H1)
                        # If L1 doesn't hit, L2 may hit
                        p2 = p_hit(i2, l2, H2)

                        # Event probabilities this pull:
                        prob_L1_hit = p1
                        prob_L2_hit = (1.0 - p1) * p2
                        prob_none = (1.0 - p1) * (1.0 - p2)

                        # --- NONE hits ---
                        i1n = min(i1 + 1, H1 - 1)
                        i2n = min(i2 + 1, H2 - 1)
                        A[row, idx(i1n, j1, i2n, j2)] -= prob_none

                        # --- L1 HIT branch ---
                        # Update L1 targeting
                        forced1 = (j1 == G1 - 1) and (G1 > 1)
                        if forced1:
                            # L1 is rate-up
                            l1_desired = l1.rateup_desired  # rate-up desired?
                            if absorb_l1 and l1_desired:
                                pass  # absorption
                            else:
                                # not absorbed: update states
                                new_j1 = 0
                                new_i1 = 0
                                # L2 resets fail counter on L1 hit (default)
                                new_i2 = 0
                                # L2 nonrate unchanged
                                A[row, idx(new_i1, new_j1, new_i2, j2)] -= prob_L1_hit
                        else:
                            # rate-up vs other
                            pr = prob_L1_hit * l1.C
                            po = prob_L1_hit * (1.0 - l1.C)

                            # rate-up outcome
                            l1_des_rateup = l1.rateup_desired
                            if absorb_l1 and l1_des_rateup:
                                pass
                            else:
                                # proceed to new state
                                new_i1 = 0
                                new_j1 = 0
                                new_i2 = 0  # reset L2 fail on L1 hit
                                A[row, idx(new_i1, new_j1, new_i2, j2)] -= pr

                            # other outcome: desired with prob W
                            # If desired via "other"
                            if absorb_l1:
                                # desired probability consumes absorption; only the not-desired portion transitions
                                po_not_des = po * (1.0 - l1.W)
                            else:
                                po_not_des = po  # none of L1 desired matters
                            # update j1 on "other not desired"
                            if G1 > 1:
                                new_j1 = min(j1 + 1, G1 - 1)
                            else:
                                new_j1 = 0
                            new_i1 = 0
                            new_i2 = 0
                            A[row, idx(new_i1, new_j1, new_i2, j2)] -= po_not_des

                        # --- L2 HIT branch ---
                        # L1 did not hit, so L1 fail counter increments; L2 resets fail counter
                        i1_after = min(i1 + 1, H1 - 1)

                        forced2 = (j2 == G2 - 1) and (G2 > 1)
                        if forced2:
                            l2_desired = l2.rateup_desired
                            if absorb_l2 and l2_desired:
                                pass
                            else:
                                new_i2 = 0
                                new_j2 = 0
                                A[row, idx(i1_after, j1, new_i2, new_j2)] -= prob_L2_hit
                        else:
                            pr2 = prob_L2_hit * l2.C
                            po2 = prob_L2_hit * (1.0 - l2.C)

                            # rate-up
                            l2_des_rateup = l2.rateup_desired
                            if absorb_l2 and l2_des_rateup:
                                pass
                            else:
                                new_i2 = 0
                                new_j2 = 0
                                A[row, idx(i1_after, j1, new_i2, new_j2)] -= pr2

                            # other
                            if absorb_l2:
                                po2_not_des = po2 * (1.0 - l2.W)
                            else:
                                po2_not_des = po2
                            if G2 > 1:
                                new_j2 = min(j2 + 1, G2 - 1)
                            else:
                                new_j2 = 0
                            new_i2 = 0
                            A[row, idx(i1_after, j1, new_i2, new_j2)] -= po2_not_des

        try:
            x = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return float("inf")
        sf1 = max(0, min(start_fail_1, H1 - 1))
        sj1 = max(0, min(start_nonrate_1, G1 - 1))
        sf2 = max(0, min(start_fail_2, H2 - 1))
        sj2 = max(0, min(start_nonrate_2, G2 - 1))
        return float(x[idx(sf1, sj1, sf2, sj2)])

    ev_l1 = solve(absorb_l1=True, absorb_l2=False)
    ev_l2 = solve(absorb_l1=False, absorb_l2=True)
    ev_any = solve(absorb_l1=True, absorb_l2=True)
    return ev_l1, ev_l2, ev_any


# -------------------------
# Template loading/resolving
# -------------------------

def load_templates_csv(path: str) -> Tuple[List[GameTemplate], Dict[str, GameTemplate]]:
    templates: List[GameTemplate] = []
    alias_map: Dict[str, GameTemplate] = {}

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            canonical = _s(row.get("canonical", ""))
            if not canonical:
                continue
            aliases_raw = _s(row.get("aliases", ""))
            aliases = [a.strip() for a in aliases_raw.split("|") if a.strip()]
            all_names = [canonical] + aliases

            l1 = LevelParams(
                B=float(row["l1_B"]),
                P=parse_opt_int(row.get("l1_P", "")),
                R=float(row.get("l1_R") or 0.0),
                H=parse_opt_int(row.get("l1_H", "")),
                C=float(row.get("l1_C") or 0.0),
                G=int(row.get("l1_G") or 1),
                W=0.0,
                rateup_desired=parse_bool(row.get("l1_rateup_desired", "")),
            )

            l2_B = parse_opt_float(row.get("l2_B", ""))
            l2 = None
            if l2_B is not None:
                l2 = LevelParams(
                    B=float(l2_B),
                    P=parse_opt_int(row.get("l2_P", "")),
                    R=float(row.get("l2_R") or 0.0),
                    H=parse_opt_int(row.get("l2_H", "")),
                    C=float(row.get("l2_C") or 0.0),
                    G=int(row.get("l2_G") or 1),
                    W=0.0,
                    rateup_desired=parse_bool(row.get("l2_rateup_desired", "")),
                )

            notes = _s(row.get("notes", ""))

            gt = GameTemplate(canonical=canonical, aliases=aliases, l1=l1, l2=l2, notes=notes)
            templates.append(gt)

            for nm in all_names:
                key = normalize_name(nm)
                if key:
                    alias_map[key] = gt

    return templates, alias_map


def resolve_template(alias_map: Dict[str, GameTemplate], name: str) -> GameTemplate:
    key = normalize_name(name)
    if key in alias_map:
        return alias_map[key]
    raise KeyError(f"Unknown template name: {name}")
