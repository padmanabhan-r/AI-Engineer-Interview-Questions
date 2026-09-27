"""Safety Q24: guardrails for a healthcare chatbot that must not diagnose."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 384, "Healthcare chatbot guardrails", "Scope set with clinicians and enforced in code at three points; emergencies take a fixed path the model never writes.")
Y = 225
F, A = PALETTE["fail"][2], PALETTE["amber"][2]
s.box(24, Y - 28, 100, 56, "Message", ["+ history"], "human")
s.arrow([(124, Y), (148, Y)], "human")
s.diamond(205, Y, 110, 90, "Emergency?", "fail", size=12)
s.arrow([(260, Y), (284, Y)], "slate")
s.text(271, Y - 12, "no", size=11, weight=600, fill="#344054")
s.diamond(348, Y, 124, 90, "Diagnosis\nrequest?", "amber", size=12)
s.arrow([(410, Y), (438, Y)], "slate")
s.text(423, Y - 12, "no", size=11, weight=600, fill="#344054")
s.box(440, Y - 30, 144, 60, "LLM", ["approved content only"], "model")
s.arrow([(584, Y), (604, Y)], "model")
s.hexagon(667, Y, 122, 60, "Output check", "amber", size=12.5)
s.arrow([(728, Y), (768, Y)], "output")
s.text(748, Y - 12, "pass", size=11, weight=600, fill=PALETTE["output"][2])
s.box(770, Y - 28, 106, 56, "Answer", ["+ sources"], "output")

s.arrow([(205, 180), (205, 144)], "fail")
s.text(215, 164, "yes", size=11, weight=600, fill=F, anchor="start")
s.box(110, 86, 190, 56, "Fixed response", ["+ human escalation"], "fail")
s.arrow([(348, 270), (348, 298)], "amber")
s.text(358, 284, "yes", size=11, weight=600, fill=A, anchor="start")
s.box(258, 300, 180, 56, "Triage reply", ["templated"], "amber")
s.arrow([(667, 195), (667, 144)], "fail")
s.text(677, 172, "fail", size=11, weight=600, fill=F, anchor="start")
s.box(575, 86, 184, 56, "Regenerate once", ["then fall back"], "amber")

s.text(670, 284, "Scope, set with clinicians", size=11.5, weight=700, fill="#344054")
s.box(465, 300, 190, 56, "Allowed", ["general info, logistics"], "output", size=13)
s.box(665, 300, 210, 56, "Forbidden", ["likely condition, dosing, labs"], "fail", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/10-ai-safety-ethics-and-responsible-ai/q24-healthcare-guardrails.svg")
