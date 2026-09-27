"""Safety Q20: layered abuse controls, scored across requests, sessions and accounts."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 480, "Layered abuse controls", "Treat abuse like fraud: score risk across requests, sessions and accounts, not one message at a time.")
Y = 140
s.pill(80, Y, 110, 40, "Request", "slate")
s.arrow([(135, Y), (173, Y)], "slate")
s.hexagon(250, Y, 150, 60, "Auth + quotas", "amber")
s.text(250, 188, "verified accounts, cost caps", size=11, fill=PALETTE["amber"][2])
s.arrow([(325, Y), (368, Y)], "compute")
s.box(370, 110, 170, 60, "Classifiers", ["per request"], "compute")
s.arrow([(540, Y), (578, Y)], "data")
s.box(580, 102, 270, 76, "Behavioral signals", ["session and account level", "repeated refusals, near-duplicates"], "data")
s.arrow([(715, 178), (715, 263)], "amber")
s.diamond(715, 310, 120, 90, "Risk\nscore", "amber", size=13)

# outcomes, fanned out to the left
s.arrow([(655, 310), (567, 250)], "output")
s.text(628, 262, "low", size=11.5, weight=600, fill=PALETTE["output"][2])
s.pill(480, 250, 170, 36, "Serve", "output")
s.arrow([(655, 310), (567, 310)], "amber", label="medium", label_dy=-10)
s.pill(480, 310, 170, 36, "Friction", "amber")
s.arrow([(655, 310), (567, 370)], "fail")
s.text(628, 358, "high", size=11.5, weight=600, fill=PALETTE["fail"][2])
s.pill(480, 370, 170, 36, "Block + review", "fail")

# feedback: confirmed abuse becomes labels
s.arrow([(395, 370), (330, 370), (330, 214), (455, 214), (455, 172)], "model", dashed=True)
s.text(320, 290, "labels", size=11.5, weight=600, fill=PALETTE["model"][2], anchor="end")
s.text(320, 306, "retrain the", size=11.5, weight=600, fill=PALETTE["model"][2], anchor="end")
s.text(320, 322, "classifiers", size=11.5, weight=600, fill=PALETTE["model"][2], anchor="end")

# enforcement ladder
s.text(40, 436, "Enforcement steps up:", size=12, weight=700, fill="#344054", anchor="start")
xs = [236, 372, 508, 644]
for i, (label, role) in enumerate([("friction", "amber"), ("warning", "amber"), ("suspension", "fail"), ("ban", "fail")]):
    s.pill(xs[i], 436, 106, 32, label, role, size=12)
    if i < 3:
        s.arrow([(xs[i] + 53, 436), (xs[i + 1] - 55, 436)], role)
s.text(708, 436, "+ legal reporting, appeals", size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/10-ai-safety-ethics-and-responsible-ai/q20-abuse-controls.svg")
