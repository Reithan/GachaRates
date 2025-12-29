# GachaRates

**GachaRates** is a small command-line tool that estimates
**how many pulls you should expect to make before you get something you want**
on a gacha banner.

It exists to answer questions like:

* *“Is this banner actually worth pulling on?”*
* *“How bad is the pity / guarantee system really?”*
* *“How much does my wishlist change the odds?”*

You don’t need to understand probability or math — the tool does that part for you.

---

## How do I use this?

You run the tool once per banner and it prints expected pull counts.

### Basic usage (recommended)

Use a **game template** and only specify what *you* want.

```bash
python banner.py rate "Genshin Impact" --W 0.25
```

This means:

* use the preset banner mechanics for that game
* assume you want 25% of the non-featured outcomes

To see available templates:

```bash
python banner.py --list-templates rate
```

---

### Two-level banners (split rate)

Some banners have two outcome levels (e.g. “high” and “low” rarity).

```bash
python banner.py split-rate "Genshin Impact" --l1_W 0.0 --l2_W 0.3
```

This prints:

* expected pulls to a desired **level 1** result
* expected pulls to a desired **level 2** result
* expected pulls to **any** desired result

---

<details>
<summary><strong>Advanced arguments (manual mode)</strong></summary>

You can bypass templates and describe a banner manually.

Single level:

```bash
python banner.py rate --B 0.02 --C 0.5 --W 0.2
```

Split level:

```bash
python banner.py split-rate \
  --l1_B 0.01 --l1_C 0.5 --l1_W 0.0 \
  --l2_B 0.05 --l2_C 0.5 --l2_W 0.3
```

**Common parameters**

* `B` – base chance per pull to hit this level
* `P` – soft pity threshold (optional)
* `R` – increase per pull after soft pity (optional)
* `H` – hard pity limit (optional)
* `C` – chance the hit is “rate-up”
* `G` – number of non-rate-ups before guarantee
* `W` – fraction of non-rate-up outcomes you want
* `rateup-desired` – whether the rate-up itself counts as desired

You can mix templates and overrides (overrides always win).

</details>

---

## How does this work? (high level)

Each pull updates a small internal state:

* how many pulls since the last hit
* whether a guarantee is active

The tool:

1. models all possible future pulls
2. tracks how those states change
3. computes how many pulls you expect until a “desired” outcome happens

If a desired outcome is **impossible**, the tool reports the expected value as infinite.

---

<details>
<summary><strong>Math details (optional)</strong></summary>

The banner is modeled as a finite Markov process with absorbing states
(“you got something you want”).

Expected pulls are computed by solving the linear system:

[
E = 1 + P \cdot E
]

Split-rate banners are handled with a joint state space where:

* one pull produces exactly one result
* the higher level takes precedence
* higher-level hits reset lower-level pity
* guarantee counters are preserved unless that level hits

If the system has no path to absorption, the matrix is singular and the EV is ∞.

</details>

---

## Contributing

You don’t need to touch the math to contribute.

The most useful contributions are:

* adding new games to `games.csv`
* adding aliases for existing games
* correcting or documenting banner assumptions

---

<details>
<summary><strong>How to add a game template</strong></summary>

Edit `games.csv` and add a row with:

* canonical game name
* aliases (pipe-separated)
* banner parameters for level 1
* optional parameters for level 2

Do **not** include wish ratios (`W`) — those are always user-defined.

Run tests after changes:

```bash
pytest
```

</details>

---

## Disclaimers

* This tool computes **expected values**, not guarantees.
* Real pulls are still random.
* Template values are approximations and may vary by banner or version.
* Use this to understand risk — not to justify bad decisions 😉
