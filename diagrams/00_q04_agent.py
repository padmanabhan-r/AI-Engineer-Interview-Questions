"""Must Know Q4: an agent is an LLM in a loop, with a controller, and errors compound over steps."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 470, "An agent is an LLM in a loop", "The model chooses the control flow; the runtime executes, caps the loop and asks before risky actions.")
s.pill(125, 108, 150, 40, "Goal", "human")
s.arrow([(125, 128), (125, 158)], "human")
s.box(40, 160, 170, 70, "LLM", ["reasons, emits a", "structured tool call"], "model")
s.arrow([(210, 195), (262, 195)], "model", label="tool call", label_dy=-12)
s.hexagon(338, 195, 148, 58, "Controller", "amber")
s.text(338, 240, "step + budget caps", size=11, fill="#7A5300")
s.text(338, 255, "approval if risky", size=11, fill="#7A5300")
s.arrow([(412, 195), (458, 195)], "amber")
s.box(460, 160, 160, 70, "Execute tool", ["typed schema: search,", "code, APIs"], "compute", size=13.5)
s.arrow([(540, 230), (540, 288)], "compute")
s.box(460, 290, 160, 62, "Observation", ["the tool's result"], "data", size=13.5)
s.arrow([(460, 321), (374, 321)], "data")
s.cylinder(310, 321, 124, 62, "Memory", "data")
s.text(310, 368, "transcript + optional long-term", size=11, fill="#0A5A51")
s.arrow([(248, 321), (150, 321), (150, 234)], "data")
s.arrow([(70, 230), (70, 390)], "output")
s.text(78, 300, "done or", size=11.5, fill="#275C1C", weight=600, anchor="start")
s.text(78, 316, "limit hit", size=11.5, fill="#275C1C", weight=600, anchor="start")
s.pill(150, 412, 190, 40, "Answer or escalate", "output", size=13)

# errors compound
s.region(652, 84, 204, 360, "Errors compound", "fail", dashed=False)
base, top = 378, 178
for n in range(1, 11):
    p = 0.95 ** n
    h = (base - top) * p
    x = 670 + (n - 1) * 17.5
    role = "fail" if n == 10 else "amber"
    col = PALETTE[role][0]
    s.add(f'<rect x="{x:.1f}" y="{base - h:.1f}" width="12" height="{h:.1f}" rx="2" fill="{col}" fill-opacity="0.75"/>')
s.add(f'<line x1="664" y1="{base}" x2="846" y2="{base}" stroke="#98A2B3" stroke-width="1"/>')
s.text(676, 178 + 200 * 0.05 - 12, "95%", size=11.5, weight=700, fill="#7A5300")
s.text(836, base - 200 * 0.95 ** 10 - 18, "≈60%", size=11.5, weight=700, fill="#8E2A23")
s.text(676, base + 14, "1", size=11, fill="#667085")
s.text(834, base + 14, "10", size=11, fill="#667085")
s.text(755, base + 14, "steps", size=11, fill="#667085")
s.text(754, 124, "95% right per step", size=11.5, fill="#8E2A23")
s.text(754, 140, "is ≈60% over 10 steps", size=11.5, fill="#8E2A23")
s.text(754, 412, "default to a fixed", size=11.5, weight=600, fill="#344054")
s.text(754, 428, "workflow when you can", size=11.5, weight=600, fill="#344054")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/00-must-know/q04-agent.svg")
