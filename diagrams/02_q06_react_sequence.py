"""Prompt Engineering Q6: ReAct as a message sequence between orchestrator, model and tool."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(860, 500, "ReAct: the model decides, your code acts", "The LLM writes a Thought and an Action; the orchestrator validates and runs it, then appends the Observation.")
L, O, T = 170, 430, 690  # lifelines: LLM, orchestrator, tool
for x, title, sub, role in [(L, "LLM", "writes Thought + Action", "model"), (O, "Orchestrator", "your code", "compute"), (T, "Tool", "search, API, DB", "amber")]:
    s.box(x - 95, 84, 190, 56, title, [sub], role, size=14)
    s.add(f'<line x1="{x}" y1="140" x2="{x}" y2="470" stroke="#98A2B3" stroke-width="1.5" stroke-dasharray="4 5"/>')

s.region(52, 200, 776, 196, "loop", "slate")
s.text(560, 364, "until: a final answer, a step limit,\nor a repeated identical action", size=11.5, fill="#344054", weight=600)

def msg(x0, x1, y, label, role, dashed=False, below=""):
    d = 6 if x1 > x0 else -6
    s.arrow([(x0 + d, y), (x1 - d, y)], role, dashed=dashed, label=label, label_dy=-11)
    if below:
        s.text((x0 + x1) / 2, y + 15, below, size=11, fill="#8E2A23", weight=600)

msg(O, L, 172, "question + tools + history", "compute")
msg(L, O, 238, "Thought + Action (tool call)", "model", dashed=True)
msg(O, T, 238, "validate, then execute", "compute")
msg(T, O, 292, "result", "amber", dashed=True, below="untrusted text: may carry injected instructions")
msg(O, L, 364, "history + Observation", "compute")
s.text((O + L) / 2, 380, "the context grows every step", size=11, fill="#667085", italic=True)
msg(L, O, 432, "Thought + Final Answer", "output", dashed=True)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/02-prompt-engineering/q06-react-sequence.svg")
