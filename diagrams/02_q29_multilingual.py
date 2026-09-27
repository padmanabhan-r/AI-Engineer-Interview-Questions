"""Prompt Engineering Q29: multilingual support as a pipeline, fixed stage by stage and rolled out per language."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 486, "Multilingual support is a pipeline problem", "Measure each language to find the stage that fails, fix that stage, then roll out one language at a time.")
stages = [("Detect language", ["of the input"], "slate", ""),
          ("Retrieval", ["multilingual embeddings", "+ keyword search that", "segments ZH / JA"], "data", "English-only embeddings"),
          ("Generation", ["explicit output language", "+ target-language", "examples for tone"], "model", "no output-language instruction"),
          ("Guardrails", ["moderation and PII", "classifiers checked", "in each language"], "amber", ""),
          ("Locale formatting", [], "output", "")]
W, G, Y = 156, 18, 150
for i, (t, lines, role, culprit) in enumerate(stages):
    x = 24 + i * (W + G)
    s.box(x, Y, W, 100, t, lines, role, size=13.5, detail=11)
    if i < len(stages) - 1:
        s.arrow([(x + W + 1, Y + 50), (x + W + G - 1, Y + 50)], role)
    if culprit:
        s.text(x + W / 2, Y - 34, "usual culprit:", size=11, fill="#8E2A23", weight=600)
        s.text(x + W / 2, Y - 18, culprit, size=11, fill="#8E2A23", weight=600)

# rollout, gated per language
s.region(24, 290, 852, 158, "ROLLOUT · one language at a time", "slate")
s.cylinder(150, 362, 170, 76, "Per-language evals", "data", size=12.5)
s.text(150, 412, "reviewed translations + native queries", size=11, fill="#0A5A51")
s.arrow([(236, 360), (338, 360)], "data")
s.hexagon(430, 360, 184, 60, "Passes its own eval?", "amber", size=12.5)
s.arrow([(522, 360), (602, 360)], "output", label="yes", label_dy=-10)
s.pill(690, 360, 184, 40, "Ship that language", "output", size=12.5)
s.text(150, 429, "usual culprit: evals only in English", size=11, fill="#8E2A23", weight=600)

s.text(450, 466, "“The model speaks French” is no evidence that your retrieval, prompts and classifiers work in French.", size=11.5, fill="#667085", italic=True)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/02-prompt-engineering/q29-multilingual.svg")
