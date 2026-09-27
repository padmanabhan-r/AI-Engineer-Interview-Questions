"""Fine-tuning Q20: domain adaptation starts from an expert eval and a RAG baseline, then fixes the gap by type."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 440, "Fine-tuning for a domain: diagnose the gap first", "Language and behavior are training problems; facts that change are a retrieval problem. Experts grade all of it.")
Y = 240
s.cylinder(86, Y, 116, 84, "Expert\neval set", "human", size=12.5)
s.arrow([(144, Y), (168, Y)], "human")
s.box(170, 204, 150, 72, "Baseline", ["strong general model", "+ RAG: often wins"], "slate", size=13.5)
s.arrow([(320, Y), (344, Y)], "slate")
s.diamond(410, Y, 128, 104, "Gap\ntype?", "amber")
# language
s.arrow([(410, 188), (410, 130), (528, 130)], "amber")
s.text(400, 158, "language", size=11.5, fill="#7A5300", weight=600, anchor="end")
s.box(530, 98, 186, 64, "Continued pre-training", ["large domain corpus"], "model", size=12.5)
s.arrow([(623, 162), (623, 206)], "model")
# behaviour
s.arrow([(474, Y), (528, Y)], "amber", label="behavior", label_dy=-11)
s.box(530, 208, 186, 64, "SFT", ["expert-reviewed examples,", "incl. abstentions"], "model", size=13.5)
s.arrow([(716, Y), (738, Y)], "model")
s.box(740, 208, 140, 64, "Preference tuning", ["experts rank"], "model", size=12.5)
s.arrow([(810, 272), (810, 322)], "model")
s.hexagon(810, 352, 150, 58, "Eval +\ncompliance", "amber", size=12)
# facts
s.arrow([(410, 292), (410, 352), (528, 352)], "amber")
s.text(400, 326, "facts", size=11.5, fill="#7A5300", weight=600, anchor="end")
s.box(530, 320, 186, 64, "Retrieval, not training", ["citations, figures from", "documents or tools"], "data", size=12.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/05-fine-tuning-and-model-adaptation/q20-domain-fine-tuning.svg")
