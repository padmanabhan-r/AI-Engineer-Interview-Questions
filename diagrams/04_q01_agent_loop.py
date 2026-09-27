"""Agents Q1: an agent is an LLM in a loop; the harness runs tools and decides when to stop."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 440, "An agent is an LLM in a loop", "The model picks the next action; the harness runs it, appends the result and re-sends everything. Use the least autonomy that works.")

# context -> LLM
s.box(36, 150, 164, 76, "Context", ["goal + tool schemas", "+ growing transcript"], "human", size=13.5)
s.arrow([(200, 188), (242, 188)], "human")
s.box(244, 150, 160, 76, "LLM call", ["stateless: sees only", "what is re-sent"], "model")

# tool call -> harness
s.arrow([(404, 188), (516, 188)], "amber", label="tool call", label_dy=-11)
s.text(460, 206, "name + JSON args", size=10.5, fill="#7A5300", mono=True)
s.box(518, 150, 186, 76, "Harness runs tool", ["validates, executes,", "enforces limits"], "amber", size=13.5)

# result appended, back into context
s.arrow([(611, 226), (611, 272)], "data")
s.box(518, 274, 186, 58, "Append result", ["to the transcript"], "data", size=13.5)
s.arrow([(518, 303), (118, 303), (118, 228)], "data")
s.text(318, 290, "next turn sees the real result", size=11.5, fill="#0A5A51", weight=600)

# exits
s.arrow([(324, 150), (324, 108), (730, 108)], "output")
s.text(527, 96, "final answer", size=11.5, fill="#275C1C", weight=600)
s.pill(790, 108, 112, 38, "Return", "output")
s.arrow([(704, 188), (732, 188)], "fail", dashed=True)
s.hexagon(790, 188, 112, 44, "limit hit", "fail", size=12.5)
s.text(790, 222, "harness stops it", size=11, fill="#8E2A23")

# who owns control flow
s.box(40, 360, 390, 58, "Workflow · single LLM call", ["your code owns the control flow; fixed steps"], "slate", size=13)
s.box(450, 360, 390, 58, "Agent", ["the model owns part of it; path depends on findings"], "model", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q01-agent-loop.svg")
