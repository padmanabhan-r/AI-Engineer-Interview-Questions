"""Agents Q20: context compaction: clear stale tool output first, summarize only if still over the threshold."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 580, "Context compaction", "Clear stale tool output first (no summarization loss); summarize older turns only if the context is still over the threshold.")
X, W = 40, 440


def bar(y, segs, label):
    s.text(X, y - 14, label, size=12.5, weight=700, fill="#344054", anchor="start")
    s.add(f'<rect x="{X}" y="{y}" width="{W}" height="28" rx="7" fill="#FFFFFF" stroke="#D0D5DD"/>')
    x = X
    for frac, role, t in segs:
        w = W * frac
        s.add(f'<rect x="{x + 1:.1f}" y="{y + 1}" width="{w - 2:.1f}" height="26" rx="6" fill="{PALETTE[role][0]}" fill-opacity="0.78"/>')
        if t:
            s.text(x + w / 2, y + 14, t, size=10.5, fill="#FFFFFF", weight=700)
        x += w


# 1 near the limit, with the trigger band
bar(112, [(0.07, "slate", "sys"), (0.37, "model", "older turns"), (0.40, "amber", "old tool output"), (0.10, "compute", "recent")], "Context near the limit")
bx0, bx1 = X + W * 0.70, X + W * 0.95
s.add(f'<path d="M{bx0},110 V104 H{bx1} V110" fill="none" stroke="#D0443A" stroke-width="1.8"/>')
s.text((bx0 + bx1) / 2, 93, "trigger: 70–95% of window", size=11, fill="#8E2A23", weight=600)

s.arrow([(200, 142), (200, 176)], "amber")
s.pill(200, 196, 290, 36, "Clear old tool results to stubs", "amber", size=12.5)
s.arrow([(200, 214), (200, 250)], "amber")

# 2 after clearing
bar(266, [(0.07, "slate", "sys"), (0.37, "model", "older turns"), (0.03, "amber", ""), (0.10, "compute", "recent")], "After clearing")
s.arrow([(200, 296), (200, 316)], "slate")
s.diamond(200, 350, 150, 64, "Still over?", "amber")
s.arrow([(275, 350), (444, 350)], "output")
s.text(360, 338, "no", size=11.5, fill="#275C1C", weight=600)
s.pill(512, 350, 140, 38, "Continue loop", "output", size=12.5)

s.arrow([(200, 382), (200, 404)], "model")
s.text(210, 393, "yes", size=11.5, fill="#3B2596", weight=600, anchor="start")
s.pill(200, 424, 290, 36, "LLM summarizes older turns", "model", size=12.5)
s.arrow([(200, 442), (200, 492)], "model")

# 3 rebuilt
bar(496, [(0.07, "slate", "sys"), (0.15, "pink", "summary"), (0.12, "compute", "last N")], "Rebuilt")
s.text(X, 540, "system + summary + last N turns verbatim; the agent then re-reads files it needs", size=11.5, fill="#667085", anchor="start")
s.arrow([(X + W, 510), (512, 510), (512, 371)], "output")

# what the summary must keep
s.region(600, 84, 272, 268, "The summary prompt demands", "pink", dashed=False)
keep = ["the goal, verbatim", "decisions and their reasons", "current state", "open problems", "exact paths, IDs and errors"]
for i, k in enumerate(keep):
    y = 132 + i * 42
    s.add(f'<rect x="616" y="{y - 15}" width="240" height="30" rx="8" fill="#FFFFFF" stroke="#D6408F" stroke-opacity="0.5"/>')
    s.text(630, y, k, size=12, fill="#8A1F58", anchor="start")
s.box(600, 372, 272, 90, "Pitfall", ["details and goal wording get lost;", "keep a notes file outside the context", "and test with forced compaction"], "fail", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q20-context-compaction.svg")
