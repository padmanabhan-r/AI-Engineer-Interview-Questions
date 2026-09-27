"""Evaluation Q26: continuous evaluation in production: score sampled traces, watch signals and drift, alert per slice."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 480, "Continuous evaluation in production", "Run the offline scorers on live traffic, alert per slice, and feed what humans confirm back into the golden set.")
MY = 220

s.pill(78, MY, 96, 40, "Traffic", "human")
s.arrow([(126, MY), (150, MY)], "human")
s.cylinder(204, MY, 104, 78, "Traces", "data")

# three signal lanes
BX = 290
s.arrow([(256, MY - 12), (270, MY - 12), (270, 128), (BX - 2, 128)], "data")
s.arrow([(256, MY), (BX - 2, MY)], "data")
s.arrow([(256, MY + 12), (270, MY + 12), (270, 312), (BX - 2, 312)], "data")
s.box(BX, 98, 136, 60, "Sampler", ["random +", "risk-weighted"], "slate", size=13)
s.arrow([(BX + 136, 128), (BX + 158, 128)], "slate")
s.box(BX + 160, 98, 150, 60, "Checks + judges", ["same scorers", "as offline"], "model", size=13)
s.box(BX, 190, 310, 60, "User signals", ["e.g. negative feedback"], "compute", size=13)
s.box(BX, 282, 310, 60, "Drift monitors", ["inputs and scores over time"], "amber", size=13)
s.text(BX + 235, 82, "deterministic checks: all traffic · judges: the sample", size=11, fill="#3B2596", weight=600)

# converge to alerts
AX = 770
for y in (128, 220, 312):
    x0 = BX + 310
    s.arrow([(x0, y), (642, y), (642, MY), (AX - 82, MY)] if y != MY else [(x0, y), (AX - 82, MY)], "fail")
s.hexagon(AX, MY, 160, 64, "Alerts per slice", "fail", size=13)
s.text(AX, MY + 46, "with CIs + minimum sample", size=11, fill="#8E2A23", weight=600)

# human review and golden set, second row right to left
s.arrow([(AX, MY + 56), (AX, 372)], "fail")
s.box(AX - 80, 374, 160, 60, "Human review", ["confirm failures"], "human", size=13)
s.arrow([(AX - 80, 404), (512, 404)], "human")
s.cylinder(452, 404, 116, 78, "Golden set", "amber", size=12.5)
s.text(380, 394, "nightly golden-set runs catch", size=11.5, fill="#7A5300", weight=600, anchor="end")
s.text(380, 412, "silent provider updates", size=11.5, fill="#7A5300", weight=600, anchor="end")

s.text(32, 460, "Minimum viable: traces, a daily groundedness check on a sample, weekly review of 50 flagged traces.", size=11.5, fill="#667085", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/09-evaluation-and-testing/q26-continuous-eval.svg")
