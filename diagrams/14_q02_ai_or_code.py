"""Behavioral Q2: AI or conventional code? A decision ladder that defaults to code."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 610, "AI or conventional code?", "Default to code. Reach for an LLM only when rules can't be written, errors are tolerable or catchable, and value beats cost.")

s.box(30, 80, 840, 44, "First: can anyone say what a correct output looks like? If not, no solution is ready.", (), "slate", size=12.5)

X = 190
qs = [(190, "Rules writable\nand maintainable?"), (290, "Input\nunstructured?"), (390, "Errors tolerable\nor catchable?"), (490, "Value exceeds\ncost?")]
for y, q in qs:
    s.diamond(X, y, 230, 80, q, "amber", size=12)
s.arrow([(X, 124), (X, 148)], "slate")
for (y0, _), (y1, _) in zip(qs, qs[1:]):
    s.arrow([(X, y0 + 40), (X, y1 - 42)], "slate")
for y, lab in [(240, "no"), (340, "yes"), (440, "yes"), (540, "yes")]:
    s.text(X + 10, y, lab, size=11.5, fill="#344054", weight=600, anchor="start")

# exits to the right
exits = [(190, "yes", "Code", "compute"), (290, "no, tabular", "Classical ML", "data"),
         (390, "no", "Human decides, AI assists", "human"), (490, "no", "Code", "compute")]
for y, lab, out, role in exits:
    s.arrow([(X + 115, y), (378, y)], role)
    s.text(342, y - 12, lab, size=11.5, fill=PALETTE[role][2], weight=600)
    s.box(380, y - 24, 210, 48, out, (), role, size=13)

s.arrow([(X, 530), (X, 552)], "model")
s.pill(X, 574, 250, 40, "LLM component with evals", "model", size=13)

# usually hybrid
s.region(620, 150, 250, 340, "Usually a hybrid", "slate", dashed=False)
steps = [("compute", "code", "validation"), ("model", "model", "the one judgement step,\ne.g. read a free-text email"),
         ("compute", "code", "permissions"), ("compute", "code", "calculations")]
y = 196
for role, who, what in steps:
    col, fill, ink = PALETTE[role]
    h = 64 if "\n" in what else 44
    s.add(f'<rect x="640" y="{y}" width="210" height="{h}" rx="10" fill="{fill}" stroke="{col}" stroke-width="1.6"/>')
    s.text(656, y + h / 2, who, size=11, fill=ink, weight=700, anchor="start", mono=True)
    s.text(708, y + h / 2, what, size=11.5, fill=ink, anchor="start")
    y += h + 12
s.text(745, 466, "the panel is testing restraint", size=11, fill="#667085", italic=True)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/14-behavioral-and-scenario-based-questions/q02-ai-or-code.svg")
