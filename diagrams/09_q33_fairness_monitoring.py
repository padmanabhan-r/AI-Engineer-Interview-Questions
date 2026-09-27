"""Evaluation Q33: continuous fairness monitoring: label-free leading indicators daily, outcome metrics monthly."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 560, "Monitoring fairness after deployment", "Leading indicators need no labels and run daily; outcome metrics wait for labels. Alert per group, then find the cause.")

s.box(32, 170, 150, 76, "Decisions", ["and inputs,", "per group"], "human", size=14)
# lane 1: leading
s.region(206, 84, 380, 110, "LEADING · no labels · daily", "compute")
s.arrow([(182, 196), (196, 196), (196, 146), (228, 146)], "compute")
s.box(230, 110, 330, 72, "Selection rate + score drift", ["per group: PSI, missing-data rates"], "compute", size=13)
# lane 2: lagging
s.region(206, 212, 380, 110, "LAGGING · needs outcomes · monthly", "model", label_pos="bl")
s.arrow([(182, 222), (196, 222), (196, 258), (228, 258)], "model")
s.pill(290, 258, 140, 34, "Delayed outcomes", "slate", size=11)
s.arrow([(360, 258), (378, 258)], "model")
s.box(380, 226, 180, 64, "TPR, FPR, calibration", ["per group"], "model", size=12.5)

# alert
AX = 690
s.arrow([(560, 258), (598, 258), (598, 209)], "model", head=False)
s.arrow([(560, 146), (598, 146), (598, 209)], "compute", head=False)
s.arrow([(598, 209), (AX - 58, 209)], "fail")
s.hexagon(AX, 209, 112, 64, "Alert", "fail", size=14)
s.text(AX, 166, "with CI + min sample", size=11, fill="#8E2A23", weight=600)
s.arrow([(AX + 56, 209), (764, 209)], "fail")
s.box(766, 172, 110, 74, "Root cause", ["then fix"], "amber", size=13)
s.arrow([(821, 246), (821, 290)], "amber")
s.pill(821, 312, 110, 40, "Re-audit", "output")
s.text(821, 346, "+ quarterly, regardless", size=11, fill="#275C1C", weight=600)

# the pitfall, drawn: overall flat while one group's error doubles
px, py, pw, ph = 80, 404, 420, 110
s.text(32, 380, "Why overall accuracy is not enough", size=12.5, weight=700, fill="#344054", anchor="start")
s.add(f'<line x1="{px}" y1="{py + ph}" x2="{px + pw}" y2="{py + ph}" stroke="#98A2B3"/>')
s.add(f'<line x1="{px}" y1="{py}" x2="{px}" y2="{py + ph}" stroke="#98A2B3"/>')
s.text(px - 8, py + ph / 2, "error\nrate", size=11, fill="#667085", anchor="end")
s.text(px + pw, py + ph + 14, "months since deployment →", size=11, fill="#667085", anchor="end")
ov, gb = PALETTE["slate"][0], PALETTE["fail"][0]
s.add(f'<path d="M{px},{py + 70} L{px + pw},{py + 68}" stroke="{ov}" stroke-width="2.5" fill="none"/>')
s.add(f'<path d="M{px},{py + 72} C{px + 160},{py + 70} {px + 280},{py + 50} {px + pw},{py + 34}" stroke="{gb}" stroke-width="2.5" fill="none"/>')
s.text(px + pw + 8, py + 68, "overall: flat", size=11.5, weight=600, fill="#344054", anchor="start")
s.text(px + pw + 8, py + 34, "one group: doubled", size=11.5, weight=600, fill="#8E2A23", anchor="start")

s.text(632, 420, "Causes to check", size=12.5, weight=700, fill="#7A5300", anchor="start")
for i, l in enumerate(["population shift", "upstream feature changes", "feedback loops, label shift", "provider updates (LLMs)"]):
    s.text(632, 444 + i * 20, "· " + l, size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/09-evaluation-and-testing/q33-fairness-monitoring.svg")
