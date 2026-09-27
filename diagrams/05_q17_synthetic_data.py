"""Fine-tuning Q17: a synthetic-data pipeline, where verification decides what survives."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 450, "Synthetic data: generate, then verify hard", "It works only with aggressive verification: the student inherits every teacher error.")
Y = 150
s.box(28, 112, 130, 76, "Seed", ["prompts, docs,", "personas"], "human", size=13.5)
s.arrow([(158, Y), (180, Y)], "human")
s.box(182, 112, 130, 76, "Teacher", ["stronger model", "generates"], "model", size=13.5)
s.arrow([(312, Y), (332, Y)], "model")
s.hexagon(404, Y, 140, 76, "Verify", "amber")
s.text(404, 204, "tests · schema · judge", size=11, fill="#7A5300")
s.arrow([(474, Y), (496, Y)], "amber")
s.box(498, 112, 124, 76, "Deduplicate", ["by embedding"], "compute", size=13.5)
s.arrow([(622, Y), (642, Y)], "compute")
s.box(644, 112, 110, 76, "Human", ["spot-check"], "human", size=13.5)
s.arrow([(754, Y), (772, Y)], "human")
s.box(774, 112, 100, 76, "Train", ["on the mix"], "output", size=13.5)
# what gets dropped, and what gets added
s.arrow([(404, 216), (404, 252)], "fail")
s.pill(404, 272, 124, 34, "rejected", "fail", size=12)
s.arrow([(560, 188), (560, 252)], "fail")
s.pill(560, 272, 124, 34, "near-copies", "fail", size=12)
s.pill(824, 272, 110, 34, "real data", "data", size=12)
s.arrow([(824, 255), (824, 190)], "data")
# the four ways to generate
s.region(24, 312, 852, 120, "WAYS TO GENERATE", "slate", dashed=False)
ways = [("Distillation", "real prompts, teacher answers,", "smaller student", "model"),
        ("Self-Instruct / Evol", "grow and harden instructions", "from a small human seed set", "human"),
        ("Rejection sampling", "keep only outputs a verifier", "accepts: tests, answer, JSON", "amber"),
        ("Preference pairs", "a judge or rule picks the", "better of two, for DPO", "output")]
for i, (k, a, b, role) in enumerate(ways):
    s.box(40 + i * 208, 342, 196, 76, k, [a, b], role, size=12.5, detail=10.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/05-fine-tuning-and-model-adaptation/q17-synthetic-data.svg")
