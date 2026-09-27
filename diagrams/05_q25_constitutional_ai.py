"""Fine-tuning Q25: Constitutional AI, a supervised critique-and-revise phase, then RL from AI feedback."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 460, "Constitutional AI: principles instead of harm labels", "The model critiques and revises its own answers against written principles; an AI judge then labels preferences for RL.")
s.box(28, 150, 140, 210, "Constitution", ["written principles,", "explicit and", "editable"], "pink", size=13.5)
# phase 1
s.region(192, 84, 684, 136, "1 · SUPERVISED", "model", label_pos="tl")
Y1 = 160
s.pill(264, Y1, 120, 44, "Red-team\nprompt", "human", size=12)
s.arrow([(324, Y1), (348, Y1)], "human")
s.box(350, 128, 150, 64, "Initial answer", ["helpful-only model"], "model", size=13)
s.arrow([(500, Y1), (528, Y1)], "model")
s.box(530, 128, 160, 64, "Critique, revise", ["against a sampled", "principle"], "amber", size=13)
s.arrow([(690, Y1), (714, Y1)], "amber")
s.box(716, 128, 144, 64, "Fine-tune", ["on the revisions"], "model", size=13)
# phase 2, right to left
s.region(192, 290, 684, 150, "2 · RL FROM AI FEEDBACK", "output", label_pos="bl")
Y2 = 348
s.arrow([(788, 192), (788, 314)], "model")
s.box(716, 316, 144, 64, "Sample pairs", ["two answers"], "slate", size=13)
s.arrow([(716, Y2), (690, Y2)], "slate")
s.hexagon(610, Y2, 160, 64, "AI judge", "amber")
s.text(610, 394, "picks the more compliant", size=11, fill="#7A5300")
s.arrow([(530, Y2), (502, Y2)], "amber")
s.box(350, 316, 150, 64, "Preference model", ["+ human helpfulness", "labels"], "compute", size=12.5)
s.arrow([(350, Y2), (326, Y2)], "compute")
s.pill(264, Y2, 120, 44, "RL on the\npolicy", "output", size=12)
# the constitution feeds both the critique and the judge
s.arrow([(168, 255), (610, 255), (610, 194)], "pink", dashed=True)
s.arrow([(610, 255), (610, 314)], "pink", dashed=True)
s.text(400, 243, "principles used by both", size=11.5, fill="#8A1F58", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/05-fine-tuning-and-model-adaptation/q25-constitutional-ai.svg")
