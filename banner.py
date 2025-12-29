#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Optional, Tuple

from gacha_math import (
    LevelParams,
    load_templates_csv,
    resolve_template,
    expected_pulls_single,
    expected_pulls_split,
)


def add_level_flags(p: argparse.ArgumentParser, prefix: str = "") -> None:
    # prefix "" for single; "1_" or "2_" for split
    dash = "--" + prefix
    p.add_argument(dash + "B", type=float, default=None)
    p.add_argument(dash + "P", type=int, default=None)
    p.add_argument(dash + "R", type=float, default=None)
    p.add_argument(dash + "H", type=int, default=None)
    p.add_argument(dash + "C", type=float, default=None)
    p.add_argument(dash + "G", type=int, default=None)
    p.add_argument(dash + "W", type=float, default=None)
    p.add_argument(dash + "rateup-desired", action="store_true", default=False)


def merge_level(base: LevelParams, args, prefix: str = "") -> LevelParams:
    # Merge CLI overrides into base
    get = lambda k: getattr(args, (prefix + k))
    B = get("B") if get("B") is not None else base.B
    P = get("P") if get("P") is not None else base.P
    R = get("R") if get("R") is not None else base.R
    H = get("H") if get("H") is not None else base.H
    C = get("C") if get("C") is not None else base.C
    G = get("G") if get("G") is not None else base.G
    W = get("W") if get("W") is not None else base.W
    rateup_desired = True if get("rateup_desired") else base.rateup_desired

    return LevelParams(B=B, P=P, R=R, H=H, C=C, G=G, W=W, rateup_desired=rateup_desired)


def main() -> None:
    ap = argparse.ArgumentParser(description="Expected value calculator for gacha-style pull processes.")
    ap.add_argument("template", nargs="?", help="Optional template name (from CSV).")
    ap.add_argument("--templates-csv", default="games.csv", help="Path to templates CSV.")
    ap.add_argument("--list-templates", action="store_true", help="List available templates and exit.")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # ---- rate ----
    ap_rate = sub.add_parser("rate", help="Single-level EV (one hit level).")
    add_level_flags(ap_rate, prefix="")

    ap_rate.add_argument("--start-fail", type=int, default=0)
    ap_rate.add_argument("--start-nonrate", type=int, default=0)

    # ---- split-rate ----
    ap_split = sub.add_parser("split-rate", help="Two-level EV (L1 + L2), including EV to any desired.")
    add_level_flags(ap_split, prefix="l1_")
    add_level_flags(ap_split, prefix="l2_")

    ap_split.add_argument("--start-fail-1", type=int, default=0)
    ap_split.add_argument("--start-nonrate-1", type=int, default=0)
    ap_split.add_argument("--start-fail-2", type=int, default=0)
    ap_split.add_argument("--start-nonrate-2", type=int, default=0)

    args = ap.parse_args()

    templates, alias_map = load_templates_csv(args.templates_csv)

    if args.list_templates:
        for t in templates:
            print(f"{t.canonical}  (aliases: {', '.join(t.aliases) if t.aliases else '-'})")
        return

    template_obj = None
    if args.template:
        try:
            template_obj = resolve_template(alias_map, args.template)
        except KeyError:
            # If the user didn't mean a template, they can omit it; we keep behavior simple:
            raise

    if args.cmd == "rate":
        # Base params: from template L1 or require B,C at least
        if template_obj is not None:
            base = template_obj.l1
        else:
            # Minimal defaults; user must supply at least B and C in this mode
            if args.B is None or args.C is None:
                raise SystemExit("In manual mode, provide --B and --C for 'rate'.")
            base = LevelParams(B=args.B, C=args.C)

        lp = merge_level(base, args, prefix="")

        ev = expected_pulls_single(lp, start_fail=args.start_fail, start_nonrate=args.start_nonrate)
        print(f"EV pulls to desired (single level): {ev}")
        return

    if args.cmd == "split-rate":
        if template_obj is not None:
            if template_obj.l2 is None:
                raise SystemExit("Selected template has no L2 parameters for split-rate.")
            base1, base2 = template_obj.l1, template_obj.l2
        else:
            # Manual requires at least B and C for both levels
            if args.l1_B is None or args.l1_C is None or args.l2_B is None or args.l2_C is None:
                raise SystemExit("In manual mode, provide --l1_B --l1_C --l2_B --l2_C for 'split-rate'.")
            base1 = LevelParams(B=args.l1_B, C=args.l1_C)
            base2 = LevelParams(B=args.l2_B, C=args.l2_C)

        l1 = merge_level(base1, args, prefix="l1_")
        l2 = merge_level(base2, args, prefix="l2_")

        ev1, ev2, ev_any = expected_pulls_split(
            l1, l2,
            start_fail_1=args.start_fail_1, start_nonrate_1=args.start_nonrate_1,
            start_fail_2=args.start_fail_2, start_nonrate_2=args.start_nonrate_2,
        )

        print(f"EV pulls to desired L1:  {ev1}")
        print(f"EV pulls to desired L2:  {ev2}")
        print(f"EV pulls to any desired: {ev_any}")
        return


if __name__ == "__main__":
    main()
