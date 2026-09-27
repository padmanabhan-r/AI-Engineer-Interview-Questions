"""System design Q44: dynamic pricing as demand forecasting plus constrained optimisation, learned through price experiments."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 560, "A dynamic pricing engine", "Forecast demand as a function of price, optimise inside hard guardrails, and learn elasticity from randomised tests.")

# row 1: forecast and optimise
s.region(24, 80, 852, 170, "FORECAST + OPTIMISE · 1M SKUs, several reprices a day", "compute")
s.cylinder(100, 146, 140, 60, "Competitor\nprices", "data", size=12)
s.cylinder(100, 214, 140, 60, "Sales, traffic,\nstock", "data", size=12)
s.arrow([(170, 146), (188, 146), (188, 166), (204, 166)], "data")
s.arrow([(170, 214), (188, 214), (188, 192), (204, 192)], "data")
s.box(206, 146, 180, 66, "Demand model", ["price, season, promos"], "model", size=13.5)
s.arrow([(386, 179), (412, 179)], "model")
s.box(414, 140, 186, 78, "Elasticity", ["per SKU or cluster;", "causal, from price tests"], "model", size=13.5)
s.arrow([(600, 179), (626, 179)], "compute")
s.box(628, 146, 226, 66, "Optimiser", ["max the objective per SKU"], "compute", size=13.5)

# row 2: gate, experiment, publish (right to left)
s.arrow([(741, 212), (741, 284)], "amber")
s.hexagon(741, 322, 250, 72, "Guardrails\nfloors · caps · max change/day\nlegal: no protected attributes", "amber", size=11.5)
s.arrow([(616, 322), (532, 322)], "compute")
s.box(314, 290, 216, 64, "Experiment layer", ["randomised price tests;", "bandits for thin history"], "compute", size=13.5)
s.arrow([(314, 322), (194, 322)], "output")
s.pill(112, 322, 162, 42, "Publish prices", "output", size=12.5)
s.arrow([(100, 300), (100, 246)], "data", dashed=True)
s.text(110, 272, "new sales data", size=11, fill="#0A5A51", weight=600, anchor="start")
s.arrow([(741, 358), (741, 408)], "human")
s.text(751, 381, "big moves", size=11.5, fill="#1F3864", weight=600, anchor="start")
s.pill(741, 430, 180, 42, "Human review", "human")

# the objective, as a picture
s.region(24, 384, 592, 156, "", "slate", dashed=False)
x0, x1, yb, yt = 60, 330, 516, 404
s.add(f'<line x1="{x0}" y1="{yb}" x2="{x1}" y2="{yb}" stroke="#98A2B3" stroke-width="1.5"/>')
s.add(f'<line x1="{x0}" y1="{yb}" x2="{x0}" y2="{yt}" stroke="#98A2B3" stroke-width="1.5"/>')
fl, cp = 128, 262
amb = PALETTE["amber"]
s.add(f'<rect x="{fl}" y="{yt}" width="{cp - fl}" height="{yb - yt}" fill="{amb[1]}" opacity="0.9"/>')
for x, lab in ((fl, "floor"), (cp, "cap")):
    s.add(f'<line x1="{x}" y1="{yt}" x2="{x}" y2="{yb}" stroke="{amb[0]}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    s.text(x, yb + 13, lab, size=10.5, fill=amb[2], weight=600)
vi = PALETTE["model"][0]
s.add(f'<path d="M78,{yb} C120,{yt - 20} 190,{yt + 2} 320,{yb - 44}" fill="none" stroke="{vi}" stroke-width="2.4"/>')
s.add(f'<circle cx="173" cy="420" r="4.5" fill="{vi}"/>')
s.text(173, 406, "p*", size=12, weight=700, fill=PALETTE["model"][2], mono=True)
s.text(78, yb + 13, "c", size=10.5, fill="#344054", mono=True)
s.text(x1, yb - 12, "price →", size=10.5, fill="#344054", anchor="end")
s.text(x0 + 6, yt - 6, "margin", size=10.5, fill="#344054", anchor="start")
s.text(356, 420, "Objective (margin)", size=12.5, weight=700, fill="#1B418C", anchor="start")
s.text(356, 446, "max over p of (p − c) · D(p)", size=12, fill="#344054", anchor="start", mono=True)
s.text(356, 474, "constant elasticity ε < −1:", size=12, fill="#344054", anchor="start")
s.text(356, 498, "p* = c · ε / (ε + 1)", size=12.5, weight=700, fill="#3B2596", anchor="start", mono=True)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q44-dynamic-pricing.svg")
