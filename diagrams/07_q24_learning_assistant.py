"""System design Q24: a personalised learning assistant: learner model, next-activity policy, and a Socratic LLM tutor."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 470, "A personalised learning assistant", "The LLM is the interface; a per-skill learner model and a deterministic answer checker make it personal and correct.")
s.pill(100, 262, 130, 44, "Student", "human")
# tutor
s.box(230, 96, 250, 76, "LLM tutor", ["hints and Socratic questions;", "full solution only after attempts"], "model", size=14)
col = PALETTE["model"][0]
s._marker(col)
s.add(f'<path d="M100,240 L100,134 L226,134" fill="none" stroke="{col}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" marker-start="url(#m{col[1:]})" marker-end="url(#m{col[1:]})"/>')
# grading and learner model
s.arrow([(165, 262), (254, 262)], "amber", label="answer", label_dy=-11)
s.hexagon(335, 262, 160, 62, "Answer checker\n(CAS)", "amber", size=12.5)
s.arrow([(415, 262), (468, 262)], "compute")
s.box(470, 224, 200, 76, "Learner model", ["mastery per skill (BKT),", "updated every answer"], "compute", size=14)
s.arrow([(670, 262), (712, 262)], "human")
s.pill(790, 262, 150, 44, "Teacher\ndashboard", "human", size=12.5)
# next activity
s.arrow([(570, 300), (570, 356)], "compute")
s.box(460, 358, 220, 76, "Next-activity policy", ["~70–85% expected success,", "plus spaced review"], "compute", size=13.5)
s.arrow([(460, 396), (100, 396), (100, 286)], "compute")
s.text(280, 384, "next item", size=11.5, fill="#1B418C", weight=600)
# item bank feeds both
s.cylinder(790, 134, 170, 72, "Item bank +\nworked solutions", "data", size=12)
s.arrow([(705, 134), (482, 134)], "data", label="grounds the tutor", label_dy=-11)
s.arrow([(875, 134), (886, 134), (886, 396), (682, 396)], "data")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q24-learning-assistant.svg")
