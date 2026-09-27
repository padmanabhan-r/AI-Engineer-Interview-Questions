"""Agents Q38: Sakana Fugu: one API call; a learned orchestrator routes to one model (fast) or writes a workflow (Ultra)."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 508, "Sakana Fugu: a multi-agent system behind one call", "A trained orchestrator picks models from a pool and decides how they collaborate; the routing is learned, not hand-written.")

s.pill(76, 262, 96, 40, "Query", "human")
s.arrow([(124, 262), (146, 262)], "human")
s.box(148, 222, 150, 80, "Fugu orchestrator", ["one API call", "learned routing"], "model", size=13)
s.cylinder(223, 392, 150, 80, "Model pool", "data", size=12.5)
s.text(223, 414, "frontier · open · specialized", size=10.5, fill="#0A5A51")
s.text(223, 450, "every worker comes from here", size=10.5, fill="#667085")
s.add(f'<path d="M298,262 H320 M320,150 V338" fill="none" stroke="{PALETTE["model"][0]}" stroke-width="2"/>')

# fast lane
s.region(340, 84, 430, 150, "Fugu (fast) · pick one", "compute")
s.arrow([(320, 150), (354, 150)], "model")
s.box(356, 116, 176, 68, "Selection head", ["scores each pool model", "without generating text"], "compute", size=13, detail=11)
bars = [0.35, 0.8, 0.5, 0.25]
for i, v in enumerate(bars):
    h = 44 * v
    col = PALETTE["output"][0] if v == max(bars) else "#98A2B3"
    s.add(f'<rect x="{546 + i * 12}" y="{172 - h:.1f}" width="9" height="{h:.1f}" rx="2" fill="{col}"/>')
s.arrow([(600, 150), (626, 150)], "compute")
s.box(628, 124, 120, 52, "Worker A", (), "output", size=13)
s.text(356, 214, "supervised on per-query model rewards, then sep-CMA-ES", size=10.5, fill="#1B418C", anchor="start")

# ultra lane
s.region(340, 254, 430, 188, "Fugu Ultra · build a workflow", "model")
s.arrow([(320, 338), (354, 338)], "model")
s.box(356, 290, 176, 96, "Writes a workflow", ["subtasks", "a worker per subtask", "what each agent sees"], "model", size=13, detail=11)
s.arrow([(532, 318), (566, 302)], "model")
s.arrow([(532, 358), (566, 374)], "model")
s.box(568, 280, 96, 44, "Worker B", (), "output", size=12.5)
s.box(568, 352, 96, 44, "Worker C", (), "output", size=12.5)
s.arrow([(664, 302), (712, 328)], "output")
s.arrow([(664, 374), (712, 348)], "output")
s.add_node(724, 338, r=14, role="output")
s.text(724, 370, "combine", size=11, fill="#275C1C", weight=600)
s.text(356, 424, "trained with GRPO", size=10.5, fill="#3B2596", anchor="start")

# answer
s.arrow([(748, 150), (830, 150), (830, 240)], "output")
s.arrow([(738, 338), (830, 338), (830, 284)], "output")
s.pill(830, 262, 90, 40, "Answer", "output", size=12.5)

s.text(30, 482, "Reported wins over single frontier models (coding, math, reasoning) are vendor-reported. Trade-off: opaque routing, tied to the vendor's pool and variable per-query cost.",
       size=10.5, fill="#667085", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q38-sakana-fugu.svg")
