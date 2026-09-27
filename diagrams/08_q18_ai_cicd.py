"""LLMOps Q18: CI/CD for AI applications: statistical eval gates, non-code triggers, progressive release."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 470, "CI/CD for AI: gates are evaluations", "Any change, not just code, runs statistical eval gates, then a canary that rolls back on its own.")
Y1, Y2 = 160, 320

# the change: four kinds of trigger
s.box(32, Y1 - 56, 170, 112, "Change", ["code", "prompt", "model", "index, tool schema"], "human", size=14)

# offline gates (row 1)
s.region(222, 92, 656, 136, "PRE-MERGE AND PRE-RELEASE GATES", "amber", label_pos="tl")
gates = [(320, "Unit + schema\ntests", "deterministic"), (510, "Smoke eval", "every PR, cheap"), (730, "Full regression", "Δ vs production")]
prev = 202
for cx, lab, sub in gates:
    s.arrow([(prev, Y1), (cx - 84, Y1)], "amber")
    s.hexagon(cx, Y1, 164, 62, lab, "amber", size=12.5 if "\n" in lab else 13)
    s.text(cx - (22 if cx == 730 else 0), Y1 + 46, sub, size=11, fill="#7A5300", weight=600)
    prev = cx + 82
# row 2 right to left
s.arrow([(768, Y1 + 31), (768, Y2 - 33)], "amber")
s.text(730, Y2 + 46, "last offline gate", size=11, fill="#8E2A23", weight=600)
s.hexagon(730, Y2, 164, 62, "Safety suite", "fail")
s.arrow([(648, Y2), (590, Y2)], "compute")
s.box(430, Y2 - 34, 158, 68, "Canary", ["with online evals"], "compute", size=14)
s.arrow([(430, Y2), (358, Y2)], "output")
s.pill(284, Y2, 144, 42, "Full rollout", "output")
s.arrow([(509, Y2 + 34), (509, 402)], "fail")
s.text(518, 380, "quality breach", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.pill(509, 422, 150, 38, "Auto rollback", "fail")

# versioned release bundle
s.add('<rect x="32" y="250" width="170" height="130" rx="12" fill="#FFFFFF" stroke="#98A2B3" stroke-dasharray="4 4"/>')
s.text(46, 268, "Release artifacts", size=12, weight=700, fill="#344054", anchor="start")
for i, l in enumerate(["container", "prompt versions", "pinned model IDs", "index snapshot", "eval report"]):
    s.text(46, 290 + i * 18, "· " + l, size=11.5, fill="#344054", anchor="start")

s.text(830, 422, "Gate on delta vs production,\nnot only absolute thresholds.", size=11.5, fill="#667085", anchor="end")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/08-llmops-and-production-ai/q18-ai-cicd.svg")
