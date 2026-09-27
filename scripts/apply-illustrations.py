"""Swap Mermaid blocks for the SVGs listed in diagrams/manifest/*.tsv, then report any Mermaid left.
Each manifest line: topics/<file>.md <TAB> question <TAB> block index in that question (1-based) <TAB> assets/...svg <TAB> alt
python3 scripts/apply-illustrations.py"""
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
jobs = defaultdict(list)
for m in sorted((ROOT / "diagrams/manifest").glob("*.tsv")):
    for line in m.read_text().splitlines():
        if not line.strip():
            continue
        f, q, k, svg, alt = (line.split("\t") + [""] * 5)[:5]
        if not (ROOT / svg).exists():
            print(f"MISSING SVG {svg} ({m.name})"); continue
        jobs[f].append((int(q), int(k), svg, alt.replace('"', "'")))

for f, items in sorted(jobs.items()):
    p = ROOT / f
    text = p.read_text()
    heads = [(int(h.group(1)), h.start()) for h in re.finditer(r"^## (\d+)\. ", text, re.M)] + [(10**9, len(text))]
    # Work from the end so earlier offsets stay valid.
    for q, k, svg, alt in sorted(items, key=lambda t: (t[0], t[1]), reverse=True):
        idx = [i for i, (n, _) in enumerate(heads) if n == q]
        if not idx:
            print(f"NO QUESTION {f} Q{q}"); continue
        a, b = heads[idx[0]][1], heads[idx[0] + 1][1]
        blocks = list(re.finditer(r"```mermaid\n.*?```\n", text[a:b], re.S))
        if len(blocks) < k:
            print(f"NO MERMAID BLOCK {k} in {f} Q{q}"); continue
        blk = blocks[k - 1]
        tag = f'<p align="center"><img src="../{svg}" alt="{alt}" width="100%"></p>\n'
        text = text[:a + blk.start()] + tag + text[a + blk.end():]
        heads = [(n, pos) for n, pos in heads]  # positions after this block shift, but we go backwards
    p.write_text(text)
    print(f"applied {len(items):3} in {f}")

left = {f.name: len(re.findall(r"^```mermaid", f.read_text(), re.M)) for f in sorted((ROOT / "topics").glob("*.md"))}
left = {k: v for k, v in left.items() if v}
print("Mermaid left:", left or "none")
