"""Evaluation Q25: building an eval set without labelled ground truth: experts review, they do not author."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 450, "Experts as reviewers, not authors", "Generate cases and draft labels cheaply; spend expert time only on disagreements, a random sample and calibration.")
Y = 150

s.cylinder(90, Y, 116, 80, "Logs + docs", "data", size=12.5)
s.arrow([(148, Y), (176, Y)], "data")
s.box(178, Y - 38, 164, 76, "Inputs", ["clustered real traffic", "+ synthetic for gaps"], "data", size=13.5)
s.arrow([(342, Y), (370, Y)], "model")
s.box(372, Y - 38, 150, 76, "Drafted labels", ["by a model;", "binary or pairwise"], "model", size=13)
s.arrow([(522, Y), (548, Y)], "amber")
s.diamond(626, Y, 156, 96, "Labellers\nagree?", "amber")
s.arrow([(704, Y), (744, Y)], "output", label="yes", label_dy=-10)
s.pill(806, Y, 120, 42, "Accepted", "output")

# expert review
s.arrow([(626, Y + 48), (626, 272)], "fail")
s.text(636, 236, "no", size=11.5, weight=600, fill="#8E2A23", anchor="start")
s.box(540, 274, 172, 72, "Expert review", ["disagreements + sample"], "human", size=14)
s.arrow([(790, Y + 21), (700, 272)], "output", dashed=True)
s.text(742, 250, "random sample\nof agreements", size=11.5, weight=600, fill="#275C1C", anchor="start")

# the result
s.cylinder(806, 356, 124, 84, "Golden set", "amber", size=12.5)
s.arrow([(850, Y + 19), (850, 312)], "output")
s.arrow([(712, 330), (742, 330)], "human")
s.text(806, 412, "+ calibrated judge", size=11.5, weight=700, fill="#7A5300")

# how the judge is calibrated
s.add('<rect x="32" y="248" width="420" height="104" rx="12" fill="#FFFFFF" stroke="#D0D5DD"/>')
s.text(48, 268, "Calibrate, then scale", size=12.5, weight=700, fill="#344054", anchor="start")
s.text(48, 292, "Calibrate a judge on 50–150 expert-reviewed items,", size=11.5, fill="#344054", anchor="start")
s.text(48, 310, "then let it scale, with audits.", size=11.5, fill="#344054", anchor="start")
s.text(48, 334, "For RAG, a question generated from a chunk comes with its label.", size=11.5, fill="#0A5A51", anchor="start")

s.text(32, 392, "Pitfall: synthetic inputs are cleaner than real ones and drafted labels share the", size=11.5, weight=600, fill="#8E2A23", anchor="start")
s.text(32, 410, "model's blind spots. The random expert sample measures that error.", size=11.5, weight=600, fill="#8E2A23", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/09-evaluation-and-testing/q25-eval-set-without-labels.svg")
