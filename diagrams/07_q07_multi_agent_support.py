"""System design Q7: multi-agent customer support, triage to specialists, policy in tool code, a path to humans."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 540, "Multi-agent customer support", "Triage hands off to a few specialists; the tools enforce policy, and a human is always one handoff away.")

s.pill(76, 256, 104, 42, "Customer", "human", size=12.5)
s.arrow([(128, 256), (150, 256)], "human")
s.hexagon(214, 256, 128, 60, "Verify\nidentity", "amber", size=12.5)
s.text(214, 306, "customer ID comes", size=11, fill="#7A5300", weight=600)
s.text(214, 321, "from the session", size=11, fill="#7A5300", weight=600)
s.arrow([(278, 256), (300, 256)], "amber")
s.box(302, 222, 120, 68, "Triage", ["agent"], "model", size=14)

# specialists
s.region(452, 86, 214, 330, "Specialists", "model", label_pos="bl")
s.text(559, 432, "own prompt, own tools", size=11.5, fill="#3B2596", weight=600)
spec = [(106, "Billing agent", "refunds up to a limit"), (196, "Orders agent", ""), (298, "Tech agent", "troubleshooting + KB")]
for y, t, d in spec:
    s.box(470, y, 178, 60, t, [d] if d else [], "model", size=13.5)
    s.arrow([(422, 256), (442, 256), (442, y + 30), (468, y + 30)], "model")

# policy in tool code, then systems
s.arrow([(648, 136), (686, 136), (686, 174), (704, 174)], "model")
s.arrow([(648, 226), (686, 226), (686, 190), (704, 190)], "model")
s.hexagon(790, 182, 172, 66, "Tools with\npolicy checks", "fail", size=12.5)
s.text(790, 120, "limits, eligibility", size=11.5, fill="#8E2A23", weight=600)
s.arrow([(790, 215), (790, 262)], "slate")
s.cylinder(790, 300, 168, 70, "", "slate")
s.text(790, 306, "CRM, billing,\norder APIs", size=12.5, fill="#344054", weight=600)

# humans
s.arrow([(648, 328), (686, 328), (686, 400), (704, 400)], "human")
s.box(706, 370, 164, 62, "Human handoff", ["with a summary"], "human", size=13.5)
s.arrow([(362, 290), (362, 490), (790, 490), (790, 434)], "human", dashed=True)
s.text(568, 480, "explicit request, 2 unresolved turns, negative sentiment", size=11.5, fill="#1F3864", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q07-multi-agent-support.svg")
