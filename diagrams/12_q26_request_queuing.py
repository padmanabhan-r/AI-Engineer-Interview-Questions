"""Infrastructure Q26: admission control and weighted fair queues in front of the model engines."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 440, "Request queuing with priorities", "Cost each request in tokens, reject early when a deadline can't be met, and share capacity fairly by class and tenant.")
Y = 150
s.pill(84, Y, 110, 42, "Request", "human")
s.arrow([(139, Y), (164, Y)], "human")
s.box(166, 112, 170, 76, "Classify", ["tenant tier + route,", "never a client flag"], "slate", size=13.5)
s.arrow([(336, Y), (360, Y)], "slate")
s.box(362, 112, 164, 76, "Estimate tokens", ["input + max_tokens,", "or historical output"], "compute", size=13)
s.arrow([(526, Y), (552, Y)], "compute")
s.diamond(630, Y, 152, 104, "Budget and\ndeadline OK?", "amber", size=12)
s.arrow([(706, Y), (740, Y)], "fail")
s.text(722, Y - 12, "no", size=11.5, fill="#8E2A23", weight=600)
s.box(742, 118, 132, 64, "429", ["+ Retry-After, fast"], "fail", size=14)
s.arrow([(630, 202), (630, 262)], "output")
s.text(640, 232, "yes", size=11.5, fill="#275C1C", weight=600, anchor="start")

# weighted fair queues, drained right to left into the engine
s.region(376, 264, 498, 156, "Weighted fair queues · fair share across tenants", "compute", dashed=False)
lanes = [("interactive", 70, "model"), ("standard", 25, "compute"), ("batch", 5, "slate")]
for i, (name, w, role) in enumerate(lanes):
    y = 304 + i * 38
    c, f, ink = PALETTE[role]
    s.add(f'<rect x="420" y="{y - 13}" width="330" height="26" rx="8" fill="#FFFFFF" stroke="{c}" stroke-width="1.3"/>')
    for k in range(6 - i * 2):
        s.add(f'<rect x="{426 + k * 22}" y="{y - 8}" width="16" height="16" rx="4" fill="{c}" fill-opacity="0.8"/>')
    s.text(742, y, name, size=11.5, fill=ink, weight=600, anchor="end")
    s.text(760, y, f"{w}%", size=12, fill=ink, weight=700, anchor="start", mono=True)
    s.arrow([(420, y), (398, y), (398, 342)], role, head=False)
s.arrow([(398, 342), (324, 342)], "compute")
s.box(28, 296, 294, 92, "Engine", ["continuous batching", "priority preemption under KV pressure", "dispatch to replicas with capacity"], "model", size=14)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/12-ai-infrastructure-and-scalability/q26-request-queuing.svg")
