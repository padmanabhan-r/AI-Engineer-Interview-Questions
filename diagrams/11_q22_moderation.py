"""Multimodal Q22: layered multimodal content moderation."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 412, "Layered multimodal moderation", "Exact matches first, per-modality classifiers next, a multimodal model for combined meaning, people for the middle.")
Y = 140
s.pill(72, Y, 100, 40, "Upload", "slate")
s.arrow([(122, Y), (150, Y)], "slate")
s.hexagon(215, Y, 126, 60, "Hash match", "amber", size=12.5)
s.arrow([(278, Y), (338, Y)], "data", label="no match", label_dy=-11)
s.box(340, 108, 162, 64, "Extract", ["OCR text, transcript,", "frames"], "data", size=14)
s.arrow([(502, Y), (530, Y)], "compute")
s.box(532, 110, 150, 60, "Classifiers", ["per modality"], "compute", size=14)
s.arrow([(607, 170), (607, 236)], "model")
s.box(522, 238, 170, 64, "Multimodal", ["policy model"], "model", size=14)

s.arrow([(215, 170), (215, 236)], "fail")
s.text(225, 203, "match", size=11.5, weight=600, fill=PALETTE["fail"][2], anchor="start")
s.box(110, 238, 160, 64, "Block + report", ["e.g. to NCMEC in the US"], "fail", size=13.5, detail=11)

s.arrow([(522, 252), (462, 252)], "output")
s.text(492, 241, "clear", size=11, weight=600, fill=PALETTE["output"][2])
s.pill(385, 252, 150, 34, "Allow or remove", "output", size=12)
s.arrow([(522, 294), (462, 294)], "human")
s.text(492, 309, "uncertain", size=11, weight=600, fill=PALETTE["human"][2])
s.pill(385, 294, 150, 34, "Human review", "human", size=12)

# why the multimodal step exists
s.region(712, 84, 164, 238, "Combined meaning", "pink", dashed=False)
s.pill(794, 136, 148, 32, "benign image", "output", size=12)
s.text(794, 164, "+", size=16, weight=700, fill="#667085")
s.pill(794, 192, 148, 32, "benign caption", "output", size=12)
s.text(794, 220, "=", size=16, weight=700, fill="#667085")
s.pill(794, 248, 148, 32, "hateful together", "fail", size=12)
s.text(794, 290, "only a model seeing", size=11, fill=PALETTE["pink"][2])
s.text(794, 306, "both catches it", size=11, fill=PALETTE["pink"][2])

s.text(36, 362, "Block high-severity categories synchronously; review the rest asynchronously, with reviewer wellbeing protections.",
       size=11.5, fill="#344054", anchor="start")
s.text(36, 384, "Crops, filters and overlays evade classifiers: retrain on evasion examples, and set thresholds per category as policy.",
       size=11.5, fill="#667085", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/11-multimodal-ai/q22-moderation.svg")
