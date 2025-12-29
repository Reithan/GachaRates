# tests/unit_tests.py
import math
import pytest

import gacha_math as gm


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# -------------------------
# Utility / parsing helpers
# -------------------------

def test_normalize_name():
    assert gm.normalize_name("Genshin Impact") == gm.normalize_name("genshin-impact")
    assert gm.normalize_name("  GenshinImpact  ") == gm.normalize_name("genshin impact")
    assert gm.normalize_name("GI") == "gi"
    assert gm.normalize_name("") == ""


def test_parse_bool():
    assert gm.parse_bool("true") is True
    assert gm.parse_bool("Yes") is True
    assert gm.parse_bool("1") is True
    assert gm.parse_bool("false") is False
    assert gm.parse_bool("0") is False
    assert gm.parse_bool("") is False
    with pytest.raises(ValueError):
        gm.parse_bool("maybe")


def test_parse_opt_int_float():
    assert gm.parse_opt_int("") is None
    assert gm.parse_opt_int("  ") is None
    assert gm.parse_opt_int("10") == 10

    assert gm.parse_opt_float("") is None
    assert gm.parse_opt_float("  ") is None
    assert gm.parse_opt_float("0.25") == 0.25


# -------------------------
# Probability primitives
# -------------------------

def test_implied_H_none_when_no_ramp_and_no_H():
    lp = gm.LevelParams(B=0.1, C=0.0, P=None, R=0.0, H=None, G=1, W=0.0, rateup_desired=False)
    assert gm.implied_H(lp) is None


def test_implied_H_matches_explicit_H():
    lp = gm.LevelParams(B=0.1, C=0.0, H=30, G=1)
    assert gm.implied_H(lp) == 30


def test_implied_H_from_linear_ramp():
    # Example: B=0.5, P=0, R=0.25
    # i_fail=0 -> p = 0.75
    # i_fail=1 -> p = 1.00 -> implies H = i_fail+1 = 2
    lp = gm.LevelParams(B=0.5, C=0.0, P=0, R=0.25, H=None, G=1)
    assert gm.implied_H(lp) == 2


def test_p_hit_soft_pity_and_clamp():
    lp = gm.LevelParams(B=0.1, C=0.0, P=2, R=0.2, H=None, G=1)
    H_eff = gm.implied_H(lp)  # should exist (eventually hits 1)
    assert H_eff is not None

    # Before P: constant B
    assert approx(gm.p_hit(0, lp, H_eff), 0.1)
    assert approx(gm.p_hit(1, lp, H_eff), 0.1)

    # At i_fail=2 (>=P): add R*(2-2+1)=0.2 => 0.3
    assert approx(gm.p_hit(2, lp, H_eff), 0.3)

    # Later should clamp to <= 1
    assert 0.0 <= gm.p_hit(100, lp, H_eff) <= 1.0


def test_p_desired_on_unforced_hit_cases():
    # If rateup_desired and C=1 => desired always on hit
    lp = gm.LevelParams(B=0.1, C=1.0, G=1, W=0.0, rateup_desired=True)
    assert approx(gm.p_desired_on_unforced_hit(lp), 1.0)

    # If rateup_desired false and W=1 => desired only if other occurs; but if C=1, other never occurs
    lp2 = gm.LevelParams(B=0.1, C=1.0, G=1, W=1.0, rateup_desired=False)
    assert approx(gm.p_desired_on_unforced_hit(lp2), 0.0)

    # Mix: C=0.25, rateup_desired True, W=0.2 => 0.25 + 0.75*0.2 = 0.25 + 0.15 = 0.40
    lp3 = gm.LevelParams(B=0.1, C=0.25, G=1, W=0.2, rateup_desired=True)
    assert approx(gm.p_desired_on_unforced_hit(lp3), 0.40)


# -------------------------
# Single-level EV
# -------------------------

def test_expected_pulls_single_pure_geometric_G1():
    # No finite H/ramp; solver uses geometric shortcut only when G==1
    # If every hit is desired: per-pull desired probability = B
    lp = gm.LevelParams(B=0.2, C=0.0, G=1, W=1.0, rateup_desired=False, H=None, P=None, R=0.0)
    ev = gm.expected_pulls_single(lp)
    assert approx(ev, 5.0, tol=1e-9)


def test_expected_pulls_single_hard_pity_small():
    # B=0.5, H=2, desired always on hit
    # i=1 => guaranteed next pull => E1=1
    # i=0 => E0 = 1 + 0.5*E1 = 1.5
    lp = gm.LevelParams(B=0.5, C=0.0, H=2, G=1, W=1.0, rateup_desired=False)
    ev = gm.expected_pulls_single(lp)
    assert approx(ev, 1.5, tol=1e-9)


def test_expected_pulls_single_guarantee_only_hit_every_pull():
    # H=1 => always hit each pull
    # rate-up desired, C=0.5, G=2 ("50/50 then guaranteed"):
    # expected hits to rate-up = 1.5, pulls per hit = 1 => EV pulls = 1.5
    lp = gm.LevelParams(B=1.0, C=0.5, H=1, G=2, W=0.0, rateup_desired=True)
    ev = gm.expected_pulls_single(lp)
    assert approx(ev, 1.5, tol=1e-9)


def test_expected_pulls_single_anchor_example():
    # Anchor from earlier discussion:
    # B=0.05, H=30, C=0.25, G=3, W=0, rateup_desired=True
    lp = gm.LevelParams(B=0.05, C=0.25, H=30, G=3, W=0.0, rateup_desired=True)
    ev = gm.expected_pulls_single(lp)
    assert abs(ev - 36.323) < 1e-3  # tolerant anchor


def test_expected_pulls_single_impossible_returns_inf_or_raises():
    # If neither rate-up nor other outcomes are desired, absorption never occurs.
    # Current implementation tends to produce inf in geometric case, or linear system may be singular.
    lp = gm.LevelParams(B=0.2, C=0.5, G=1, W=0.0, rateup_desired=False, H=None, P=None, R=0.0)
    ev = gm.expected_pulls_single(lp)
    assert math.isinf(ev)


# -------------------------
# Split-level EV
# -------------------------

def test_expected_pulls_split_L1_always_hits_blocks_L2():
    # L1 always hits => L2 never occurs
    l1 = gm.LevelParams(B=1.0, C=0.5, H=1, G=1, W=0.0, rateup_desired=True)
    l2 = gm.LevelParams(B=1.0, C=0.5, H=1, G=1, W=1.0, rateup_desired=False)

    ev_l1, ev_l2, ev_any = gm.expected_pulls_split(l1, l2)
    assert approx(ev_l1, 2.0, tol=1e-9)  # per pull: desired prob = C=0.5 => EV=2
    assert math.isinf(ev_l2) or ev_l2 > 1e9
    assert approx(ev_any, ev_l1, tol=1e-9)


def test_expected_pulls_split_any_desired_matches_hand_simple_case():
    # Small finite pity: both levels hit with B=0.5, H=2, and desired always on hit for both.
    # On a pull:
    #   L1 hit prob depends on i1; L2 only checked if L1 miss.
    # This is small enough that computed EV should be finite and reasonable; we mainly sanity-check:
    # EV(any) <= min(EV(L1), EV(L2)) is NOT guaranteed, but EV(any) should be <= both in this setup
    # because both are "desired always on hit", and any-hit includes both hit types.
    l1 = gm.LevelParams(B=0.5, C=0.0, H=2, G=1, W=1.0, rateup_desired=False)
    l2 = gm.LevelParams(B=0.5, C=0.0, H=2, G=1, W=1.0, rateup_desired=False)

    ev_l1, ev_l2, ev_any = gm.expected_pulls_split(l1, l2)
    assert ev_any <= ev_l1 + 1e-9
    assert ev_any <= ev_l2 + 1e-9
    assert ev_any > 0.0
