"""LLM Fundamentals Q59: layered conversation memory assembled by a token budgeter every turn."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 490, "Long conversations: layered memory, rebuilt every turn", "“Loses context after 10 turns” is a truncation choice; a token budgeter assembles each layer instead.")
src = [(110, "System prompt", "slate"), (168, "Fact store", "data"), (226, "Rolling summary", "model"),
       (284, "Recent turns, verbatim", "human"), (342, "Retrieved old turns", "compute")]
for y, lab, role in src:
    s.pill(140, y, 200, 40, lab, role, size=12.5)
    s.arrow([(240, y), (296, 214 + (y - 226) * 0.25)], role)
s.box(298, 176, 160, 76, "Context builder", ["fills a token budget", "every turn"], "amber", size=13.5)
s.arrow([(458, 214), (492, 214)], "amber")
s.box(494, 180, 110, 68, "LLM", ["answers"], "model")
s.arrow([(604, 214), (640, 214)], "model")
s.box(642, 176, 214, 76, "Memory writer", ["updates facts and summary,", "appends the transcript"], "output", size=13.5)
s.arrow([(749, 252), (749, 452), (226, 452)], "output")
s.cylinder(140, 446, 170, 58, "Raw transcript", "data", size=12.5)
s.arrow([(140, 416), (140, 364)], "compute")
s.text(150, 392, "search", size=11, fill="#1B418C", anchor="start")
s.text(236, 470, "kept as the source of truth", size=11, fill="#0A5A51", anchor="start")

# what the builder sends
segs = [("system", "slate", 70), ("facts", "data", 56), ("summary", "model", 72), ("retrieved", "compute", 76), ("recent turns", "human", 116)]
x = 318
s.text(318, 290, "What one turn sends", size=12, weight=700, fill="#344054", anchor="start")
for lab, role, w in segs:
    col, fill, ink = PALETTE[role]
    s.add(f'<rect x="{x}" y="304" width="{w - 3}" height="32" rx="6" fill="{fill}" stroke="{col}" stroke-width="1.4"/>')
    s.text(x + (w - 3) / 2, 321, lab, size=11, weight=600, fill=ink)
    x += w
s.add('<path d="M318,344 V350 H385 V344" fill="none" stroke="#667085" stroke-width="1.4"/>')
s.text(318, 364, "stable prefix: prompt caching keeps it cheap", size=11, fill="#667085", anchor="start")
s.box(560, 384, 180, 56, "Summary drift", ["test 30–50-turn scripts", "that plant facts early"], "pink", size=12.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q59-long-conversation.svg")
