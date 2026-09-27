"""Safety Q36: an appeals process for automated denials, as a sequence plus the loop around it."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 530, "An appeals process with teeth", "Notice with reasons, a one-step contest, a reviewer with real authority, and overturns fed back into evaluation.")
X = {"U": 110, "S": 390, "H": 650}
for k, name, role, w in [("U", "User", "human", 110), ("S", "System", "compute", 120), ("H", "Human reviewer", "pink", 170)]:
    s.add(f'<line x1="{X[k]}" y1="126" x2="{X[k]}" y2="420" stroke="#C9CED8" stroke-width="1.5" stroke-dasharray="4 4"/>')
    s.pill(X[k], 108, w, 36, name, role, size=12.5)

def msg(a, b, y, label, role):
    d = 4 if X[b] > X[a] else -4
    s.arrow([(X[a] + d, y), (X[b] - 1.5 * d, y)], role, label=label)

msg("S", "U", 160, "Denied + reasons + appeal link", "compute")
msg("U", "S", 208, "Appeal + corrections", "human")
msg("S", "H", 256, "Case file + decision record", "compute")
s.arrow([(X["H"] + 4, 292), (X["H"] + 44, 292), (X["H"] + 44, 324), (X["H"] + 8, 324)], "pink")
s.text(X["H"] + 54, 300, "own assessment first,", size=11.5, weight=600, fill=PALETTE["pink"][2], anchor="start")
s.text(X["H"] + 54, 316, "then the model's output", size=11.5, weight=600, fill=PALETTE["pink"][2], anchor="start")
msg("H", "S", 360, "Uphold or overturn + reason", "pink")
msg("S", "U", 404, "Outcome + next step", "compute")

s.box(30, 446, 262, 64, "Legal anchors", ["GDPR Art. 22(3) · AI Act Art. 86", "DSA complaint handling"], "human", size=13.5)
s.box(309, 446, 262, 64, "Close the loop", ["overturned cases → eval examples", "overturn rate by reason and group"], "output", size=13.5)
s.box(588, 446, 262, 64, "Warning sign", ["overturn rate near zero:", "reviewers rubber-stamp the model"], "fail", size=13.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/10-ai-safety-ethics-and-responsible-ai/q36-appeals.svg")
