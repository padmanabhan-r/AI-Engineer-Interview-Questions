"""Build README.md from topics/*.md: one row per topic with its question count. python3 scripts/build-readme.py"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows, total = [], 0
for f in sorted((ROOT / "topics").glob("*.md")):
    text = f.read_text()
    title = re.search(r"^# (.+)$", text, re.M).group(1).strip()
    n = len(re.findall(r"^## \d+\. ", text, re.M))
    total += n
    rows.append(f"| {len(rows) + 1} | [{title}](topics/{f.name}) | {n} |")

README = f"""# AI Engineer Interview Questions

{total} AI engineering interview questions across {len(rows)} topics, each answered so that someone new to the
topic can follow it: the answer in plain English first, then the idea with a small example, how it works step by
step with every term defined, the formula read out symbol by symbol where there is one, and a figure to look at.

## Topics

| # | Topic | Questions |
|---|---|---|
{chr(10).join(rows)}

## How to use it

- Read the plain-English answer, then try to explain the idea out loud before reading how it works.
- Follow the figure alongside the text; each answer says what to look at in it.
- Anything that changes fast (model names, versions, prices, regulations) is dated; check it before you quote it.
- Numbers are labelled as a typical range, a published result or a rule of thumb; scenario numbers are illustrative.

## How it is built

Figures are hand-built SVGs (one script each in `diagrams/`, drawn with `scripts/svg.py`), charts are Vega-Lite
specs in `charts/` rendered by `scripts/vl2svg.mjs`, and formulas are LaTeX. `scripts/check.py` and
`scripts/check-math.mjs` check the questions, links, lengths and every formula.
"""
(ROOT / "README.md").write_text(README)
print(f"README: {len(rows)} topics, {total} questions")
