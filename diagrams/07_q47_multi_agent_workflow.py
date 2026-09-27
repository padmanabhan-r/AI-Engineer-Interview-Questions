"""System design Q47: a multi-agent workflow engine: an explicit plan graph, typed shared state, a reviewer and a human gate, on a durable runtime."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 462, "A multi-agent workflow engine", "An explicit graph, not free-form chat: specialists write typed state, a reviewer checks it, a human approves.")

s.region(24, 78, 852, 364, "DURABLE RUNTIME · checkpoint every step · retries · step, token and time budgets", "slate")

s.pill(84, 196, 100, 42, "Task", "human")
s.arrow([(134, 196), (160, 196)], "compute")

# orchestrator with its plan graph drawn inside
st, fi, ink = PALETTE["compute"]
s.add(f'<rect x="162" y="118" width="230" height="160" rx="12" fill="{fi}" stroke="{st}" stroke-width="1.8"/>')
s.add(f'<rect x="162" y="118" width="6" height="160" rx="3" fill="{st}"/>')
s.text(280, 142, "Orchestrator", size=14, weight=700, fill=ink)
s.text(280, 162, "plans the task as a graph", size=11.5, fill=ink, opacity=0.85)
nodes = {"a": (208, 222), "b": (280, 196), "c": (280, 250), "d": (352, 222)}
for u, v in (("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")):
    (x0, y0), (x1, y1) = nodes[u], nodes[v]
    s.add(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}" stroke="{st}" stroke-width="1.6" opacity="0.7"/>')
for k, (x, y) in nodes.items():
    s.add(f'<circle cx="{x}" cy="{y}" r="9" fill="#FFFFFF" stroke="{st}" stroke-width="2"/>')
s.text(280, 222, "parallel", size=10, fill=ink, opacity=0.8)

# specialists
s.arrow([(392, 158), (438, 158)], "model")
s.arrow([(392, 236), (438, 236)], "model")
s.box(440, 128, 196, 58, "Specialist agent A", ["scoped tools + context"], "model", size=13.5)
s.box(440, 206, 196, 58, "Specialist agent B", ["scoped tools + context"], "model", size=13.5)
s.arrow([(636, 158), (668, 158)], "data")
s.arrow([(636, 236), (668, 236)], "data")
s.cylinder(754, 197, 170, 100, "Typed shared\nstate", "data")
s.text(744, 290, "schema per handoff", size=11, fill="#0A5A51", weight=600, anchor="end")

# review, approve, finish (right to left)
s.arrow([(754, 247), (754, 346)], "data")
s.box(654, 348, 200, 60, "Reviewer agent", ["checks outputs"], "model", size=13.5)
s.arrow([(654, 378), (578, 378)], "amber")
s.hexagon(480, 378, 192, 58, "Human approval", "amber", size=13)
s.arrow([(384, 378), (262, 378)], "output")
s.pill(200, 378, 120, 42, "Result", "output")

# rework loop
s.arrow([(690, 348), (690, 312), (280, 312), (280, 280)], "fail", dashed=True)
s.text(485, 300, "rework", size=11.5, fill="#8E2A23", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q47-multi-agent-workflow.svg")
