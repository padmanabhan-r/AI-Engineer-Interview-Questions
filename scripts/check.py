"""Check every topic file: questions verbatim and in order (against scripts/questions.json), TOC anchors
resolve, every image exists, code fences balance, and prose length stays within the cap.
python3 scripts/check.py"""
import json, re, statistics, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = json.loads((ROOT / "scripts/questions.json").read_text())
CAP, CAP_DESIGN = 350, 420
bad = 0

def slug(h: str) -> str:  # GitHub's heading anchor rule
    h = h.strip().lower()
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")

def prose_words(body: str) -> int:
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    body = re.sub(r"<p[^>]*>\s*<img[^>]*>\s*</p>", "", body)  # illustrations are not prose
    lines = [l for l in body.splitlines() if not l.lstrip().startswith("|") and l.strip() != "---"]
    return len(" ".join(lines).split())

for name, qs in QUESTIONS.items():
    out = ROOT / "topics" / name
    if not out.exists():
        print(f"MISSING  {name}"); bad += 1; continue
    o = out.read_text()
    heads = re.findall(r"^## (\d+)\. (.+)$", o, re.M)
    problems = []
    if [h for _, h in heads] != qs:
        diff = next((i for i, (a, b) in enumerate(zip([h for _, h in heads], qs)) if a != b), min(len(heads), len(qs)))
        problems.append(f"questions differ at #{diff + 1} ({len(heads)} vs {len(qs)})")
    if [int(n) for n, _ in heads] != list(range(1, len(heads) + 1)):
        problems.append("numbering not 1..n")
    anchors = {slug(f"{n}. {h}") for n, h in heads}
    for a in re.findall(r"\]\(#([^)]+)\)", o):
        if a not in anchors: problems.append(f"TOC anchor not found: #{a[:60]}")
    for img in re.findall(r'<img src="\.\./([^"]+)"', o):
        if not (ROOT / img).exists(): problems.append(f"missing image {img}")
    if o.count("```") % 2: problems.append("unbalanced code fences")
    cap = CAP_DESIGN if name.startswith("07-") else CAP
    bodies = re.split(r"^## \d+\. .*$", o, flags=re.M)[1:]
    wc = [prose_words(b) for b in bodies]
    over = [i + 1 for i, w in enumerate(wc) if w > cap]
    if over: problems.append(f"over {cap} words: {over}")
    status = "ok  " if not problems else "FAIL"
    bad += bool(problems)
    print(f"{status} {name:48} {len(heads):3} q  median {int(statistics.median(wc)) if wc else 0:3}  max {max(wc) if wc else 0:3}")
    for p in problems[:6]: print("       -", p)
sys.exit(1 if bad else 0)
