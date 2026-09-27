"""Prompt Engineering Q17: the prompt evaluation loop, and why the ship gate looks at slices."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 440, "Evaluating a prompt: a loop, not a glance", "Fix a set before tuning, change one thing at a time, and ship only when no slice gets worse.")
Y = 166
s.cylinder(92, Y, 124, 84, "Eval set", "data")
s.text(92, Y - 58, "50–200 cases to start", size=11, fill="#0A5A51")
s.arrow([(154, Y), (188, Y)], "data")
s.box(190, Y - 30, 140, 60, "Run prompt vN", ["in CI, every change"], "compute", size=13.5)
s.arrow([(330, Y), (366, Y)], "compute")
s.box(368, Y - 52, 198, 104, "Score, cheapest first", ["1. code: schema, exact match", "2. LLM judge + rubric", "3. human labels"], "amber", size=13.5)

# improve loop
s.arrow([(467, Y + 52), (467, 286)], "fail")
s.box(392, 288, 150, 58, "Cluster failures", ["fix the largest first"], "fail", size=13.5)
s.arrow([(392, 317), (338, 317)], "human")
s.box(184, 288, 152, 58, "Change one thing", ["then rerun all"], "human", size=13.5)
s.arrow([(260, 288), (260, Y + 32)], "human")

# production feeds the set
s.pill(92, 317, 140, 40, "Production\nfailures", "slate", size=12)
s.arrow([(92, 296), (92, Y + 44)], "slate", dashed=True)

# ship gate
s.arrow([(566, Y), (596, Y)], "amber")
s.hexagon(676, Y, 156, 60, "No slice\nregresses?", "amber", size=12.5)
s.arrow([(754, Y), (794, Y)], "output", label="yes", label_dy=-10)
s.pill(836, Y, 76, 40, "Ship", "output")

# slice vs average
s.region(588, 244, 288, 170, "The average can hide a slice", "slate", dashed=False)
zero = 732
s.add(f'<line x1="{zero}" y1="276" x2="{zero}" y2="394" stroke="#98A2B3" stroke-width="1.5"/>')
st, fi, ink = PALETTE["output"]
s.text(zero - 10, 300, "average", size=11.5, fill=ink, weight=600, anchor="end")
s.add(f'<rect x="{zero}" y="290" width="{2 * 8}" height="20" rx="3" fill="{st}" fill-opacity="0.8"/>')
s.text(zero + 24, 300, "+2 points", size=11.5, fill=ink, weight=700, anchor="start")
st, fi, ink = PALETTE["fail"]
s.text(zero + 10, 350, "slice that matters", size=11.5, fill=ink, weight=600, anchor="start")
s.add(f'<rect x="{zero - 15 * 8}" y="340" width="{15 * 8}" height="20" rx="3" fill="{st}" fill-opacity="0.8"/>')
s.text(zero - 60, 376, "−15 points", size=11.5, fill=ink, weight=700)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/02-prompt-engineering/q17-prompt-eval-loop.svg")
