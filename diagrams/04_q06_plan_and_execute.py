"""Agents Q6: Plan-and-Execute: a planner writes the whole plan, an executor runs one step at a time, a replanner revises."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 470, "Plan-and-Execute", "Plan the whole task up front, execute step by step, replan only on surprise. Best when the task's structure is knowable in advance.")

s.pill(88, 130, 110, 40, "Task", "human")
s.arrow([(143, 130), (170, 130)], "human")
s.box(172, 100, 150, 60, "Planner", ["strongest model"], "model")
s.arrow([(322, 130), (368, 130)], "model")

# the plan, as an inspectable list
s.region(370, 88, 162, 176, "Plan", "model", dashed=False)
rows = [("output", "step 1  ✓ done"), ("output", "step 2  ✓ done"), ("compute", "step 3  ◂ now"), ("slate", "step 4")]
for i, (role, label) in enumerate(rows):
    y = 128 + i * 34
    col, fill, ink = PALETTE[role]
    s.add(f'<rect x="386" y="{y - 13}" width="130" height="26" rx="7" fill="{fill}" stroke="{col}" stroke-opacity="0.7"/>')
    s.text(398, y, label, size=11.5, fill=ink, anchor="start", weight=600 if role == "compute" else 400)

# executor on the current step
s.arrow([(532, 196), (578, 196)], "compute")
s.box(580, 160, 172, 72, "Executor: step i", ["small model, ReAct loop", "or plain code"], "compute", size=13.5)
s.arrow([(666, 232), (666, 266)], "compute")
s.text(678, 249, "sees only its step + inputs", size=11, fill="#1B418C", anchor="start")
s.diamond(666, 306, 128, 76, "On track?", "amber")

# next step: back to the plan
s.arrow([(602, 306), (552, 306), (552, 230), (534, 230)], "compute")
s.text(566, 322, "next step", size=11.5, fill="#1B418C", weight=600)

# surprise: replan the rest
s.arrow([(666, 344), (666, 382)], "amber")
s.text(676, 362, "surprise", size=11.5, fill="#7A5300", weight=600, anchor="start")
s.box(580, 384, 172, 60, "Replanner", ["revises remaining steps"], "amber", size=13.5)
s.arrow([(580, 414), (450, 414), (450, 266)], "amber")
s.text(515, 402, "rewrites the rest", size=11.5, fill="#7A5300", weight=600)

# done
s.arrow([(730, 306), (770, 306)], "output")
s.text(750, 292, "done", size=11.5, fill="#275C1C", weight=600)
s.pill(822, 306, 100, 40, "Answer", "output")

# variants
s.text(40, 292, "Variants", size=12.5, weight=700, fill="#344054", anchor="start")
s.box(36, 306, 384, 56, "ReWOO", ["placeholders; tools run without re-calling the planner"], "slate", size=13)
s.box(36, 376, 384, 56, "LLMCompiler", ["emits a DAG; independent steps run in parallel"], "slate", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q06-plan-and-execute.svg")
