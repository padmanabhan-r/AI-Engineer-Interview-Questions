"""Safety Q34: a runaway feedback loop, and where each fix from the answer cuts it."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 444, "A runaway feedback loop", "Decisions choose which outcomes get recorded, and those outcomes retrain the model; exploration is the only proof you are out.")
W, H = 180, 56
nodes = {"top": (280, 124, "Model predicts", ["high risk in area A"], "model"),
         "right": (445, 256, "More scrutiny", ["in area A"], "human"),
         "bottom": (280, 388, "More incidents", ["recorded in A"], "amber"),
         "left": (115, 256, "Training data", ["over-represents A"], "data")}
for cx, cy, t, l, r in nodes.values():
    s.box(cx - W / 2, cy - H / 2, W, H, t, l, r, size=13.5)
s.arrow([(370, 124), (430, 124), (445, 170), (445, 226)], "human", curve=True)
s.arrow([(445, 286), (445, 345), (430, 388), (372, 388)], "amber", curve=True)
s.arrow([(190, 388), (130, 388), (115, 345), (115, 286)], "data", curve=True)
s.arrow([(115, 226), (115, 170), (130, 124), (188, 124)], "model", curve=True)
s.text(280, 246, "runaway", size=13, weight=700, fill=PALETTE["fail"][2])
s.text(280, 266, "feedback loop", size=13, weight=700, fill=PALETTE["fail"][2])

s.region(572, 82, 304, 344, "Break it on the data side", "slate", dashed=False)
rows = [("human", "Exploration", "cuts: decisions", "randomize a small, governed share"),
        ("human", "Holdout", "cuts: decisions", "a slice outside the model's influence"),
        ("data", "Propensity weights", "cuts: training", "log P(decision); IPW, reject inference"),
        ("amber", "Independent labels", "cuts: outcomes", "victim reports, other lenders' data")]
for i, (role, title, tag, detail) in enumerate(rows):
    y = 114 + i * 76
    col, fill, ink = PALETTE[role]
    s.add(f'<rect x="588" y="{y}" width="272" height="64" rx="10" fill="{fill}" stroke="{col}" stroke-width="1.4"/>')
    s.add(f'<rect x="588" y="{y}" width="5" height="64" rx="2.5" fill="{col}"/>')
    s.text(604, y + 21, title, size=13, weight=700, fill=ink, anchor="start")
    s.text(848, y + 21, tag, size=10.5, weight=600, fill=ink, anchor="end", opacity=0.8)
    s.text(604, y + 44, detail, size=11.5, fill=ink, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/10-ai-safety-ethics-and-responsible-ai/q34-feedback-loop.svg")
