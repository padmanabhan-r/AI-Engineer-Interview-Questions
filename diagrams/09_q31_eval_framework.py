"""Evaluation Q31: setting up an eval framework from scratch: start from error analysis, then automate and extend to production."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 440, "An eval framework from scratch", "Start from error analysis, not a metric list: read real outputs, turn each failure mode into a check.")
W, H, Y1, Y2 = 148, 74, 116, 262
X = [32, 204, 376, 548, 720]
cx = [x + W / 2 for x in X]

s.region(360, 90, 520, 280, "the loop", "amber", label_pos="br")

s.box(X[0], Y1, W, H, "Define done", ["what counts", "as correct"], "human", size=13)
s.box(X[1], Y1, W, H, "Collect inputs", ["50–100 real ones"], "data", size=13)
s.box(X[2], Y1, W, H, "Read outputs", ["tag the", "failure modes"], "amber", size=13)
s.box(X[3], Y1, W, H, "One criterion", ["binary, one per", "failure mode"], "amber", size=13)
s.box(X[4], Y1, W, H, "Checks, judges", ["code for format,", "judges for semantics"], "model", size=13)
for i in range(4):
    s.arrow([(X[i] + W, Y1 + H / 2), (X[i + 1] - 2, Y1 + H / 2)], "slate" if i < 2 else "amber")

s.box(X[4], Y2, W, H, "Calibrate judges", ["against human", "labels"], "human", size=13)
s.box(X[3], Y2, W, H, "CI gate", ["per slice"], "compute", size=13)
s.box(X[2], Y2, W, H, "Production", ["same scorers", "on live traffic"], "output", size=13)
s.arrow([(cx[4], Y1 + H), (cx[4], Y2 - 2)], "model")
s.arrow([(X[4], Y2 + H / 2), (X[3] + W + 2, Y2 + H / 2)], "human")
s.arrow([(X[3], Y2 + H / 2), (X[2] + W + 2, Y2 + H / 2)], "compute")
s.arrow([(cx[2], Y2), (cx[2], Y1 + H + 2)], "amber", width=2.4)
s.text(cx[2] + 10, (Y1 + H + Y2) / 2, "production failures\nget read and tagged", size=11.5, weight=600, fill="#7A5300", anchor="start")

# starter kit
s.add('<rect x="32" y="262" width="300" height="118" rx="12" fill="#FFFFFF" stroke="#D0D5DD"/>')
s.text(48, 282, "Enough to start", size=12.5, weight=700, fill="#344054", anchor="start")
for i, (lab, mono) in enumerate([("a JSONL file of cases", False), ("a runner", False), ("a results table", False)]):
    s.text(48, 306 + i * 20, "· " + lab, size=11.5, fill="#344054", anchor="start", mono=mono)
s.text(48, 364, "Version the dataset, prompts and judge.", size=11, fill="#667085", anchor="start")

s.text(450, 410, "Pitfall: buying a platform and adopting its default metrics before doing error analysis.", size=12, weight=600, fill="#8E2A23")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/09-evaluation-and-testing/q31-eval-framework.svg")
