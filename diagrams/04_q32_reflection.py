"""Agents Q32: reflection works when an external verifier says what failed; cap the rounds."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 500, "Reflection: generate, verify, reflect, retry", "It helps when an external check says what failed. Each round roughly doubles calls for that step, so cap it at two or three.")

s.box(50, 110, 180, 66, "Generate", ["output or trajectory"], "model")
s.arrow([(230, 143), (318, 143)], "model")
s.hexagon(420, 143, 200, 64, "Verify", "amber", size=14)
s.text(420, 190, "tests · validator · critic", size=11.5, fill="#7A5300", weight=600)
s.arrow([(520, 143), (602, 143)], "output")
s.text(561, 131, "pass", size=11.5, fill="#275C1C", weight=600)
s.pill(662, 143, 116, 40, "Return", "output")

s.arrow([(420, 202), (420, 250)], "fail")
s.text(430, 226, "fail + feedback", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.box(330, 252, 180, 64, "Reflect", ["what to change"], "amber")
s.arrow([(330, 284), (218, 284)], "pink")
s.cylinder(140, 284, 150, 80, "Lesson in memory", "pink", size=12.5)
s.arrow([(140, 244), (140, 178)], "pink")
s.text(150, 212, "next attempt reads it", size=11, fill="#8A1F58", anchor="start")

s.box(566, 226, 304, 96, "No external signal?", ["Huang et al. (2023): intrinsic", "self-correction often failed to", "improve reasoning, sometimes hurt it"], "fail", size=13, detail=11)

# variants
s.text(30, 356, "Variants", size=12.5, weight=700, fill="#344054", anchor="start")
s.box(30, 370, 270, 90, "Self-Refine", ["Madaan et al., 2023", "generate, self-feedback, refine", "for a few rounds"], "model", size=13, detail=11)
s.box(315, 370, 270, 90, "Reflexion", ["Shinn et al., 2023", "after an external failure, write a", "verbal lesson to episodic memory"], "pink", size=13, detail=11)
s.box(600, 370, 270, 90, "Evaluator-optimizer", ["a separate critic prompt or model", "scores against a rubric"], "amber", size=13, detail=11)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q32-reflection.svg")
