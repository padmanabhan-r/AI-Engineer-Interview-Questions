"""Agents Q27: indirect prompt injection: separate the reader from the privileged planner, taint what it returns, gate sinks."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 590, "Contain injection by design, not detection", "The model that reads untrusted content holds no tools; its outputs are tainted values that policy keeps away from sensitive sinks.")

# trusted side
s.region(30, 84, 440, 186, "TRUSTED · privileged", "output")
s.pill(110, 152, 148, 40, "Trusted request", "human", size=12)
s.arrow([(184, 152), (206, 152)], "human")
s.box(208, 118, 228, 68, "Planner LLM", ["sees only the request; has tools"], "model")
s.arrow([(322, 186), (322, 206)], "model")
s.box(208, 208, 228, 44, "Plan as code", (), "slate", size=13.5)

# untrusted side
s.region(30, 300, 440, 170, "UNTRUSTED · quarantined", "fail")
s.cylinder(106, 392, 118, 84, "Content", "fail", size=12.5)
s.text(106, 412, "emails, web, docs", size=10.5, fill="#8E2A23")
s.arrow([(166, 392), (206, 392)], "fail")
s.box(208, 358, 228, 68, "Quarantined LLM", ["reads the content, has no tools"], "pink")

# policy
s.arrow([(436, 230), (574, 230), (574, 278)], "slate")
s.arrow([(436, 392), (540, 392), (540, 346)], "pink")
s.text(552, 392, "typed, tainted values", size=11, fill="#8A1F58", weight=600, anchor="start")
s.hexagon(574, 312, 196, 64, "Interpreter + policy\n(CaMeL taint tracking)", "amber", size=12.5)

s.add(f'<path d="M672,312 H690" fill="none" stroke="{PALETTE["slate"][0]}" stroke-width="2"/>')
s.arrow([(690, 312), (690, 204), (710, 204)], "output")
s.text(698, 256, "allowed", size=11.5, fill="#275C1C", weight=600, anchor="start")
s.box(712, 174, 158, 60, "Tool calls", ["run as planned"], "output", size=13.5)
s.arrow([(690, 312), (690, 432), (710, 432)], "fail")
s.text(698, 360, "tainted →", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.text(698, 376, "sensitive sink", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.box(712, 402, 158, 60, "Block or ask", ["a human, exact params"], "fail", size=13.5)

# close the channels
s.region(30, 494, 840, 76, "Close the exfiltration channels", "slate", dashed=False)
chips = [("egress allowlist", 190), ("never auto-render model images", 260), ("least privilege: no send", 196)]
x = 48
for label, w in chips:
    s.add(f'<rect x="{x}" y="522" width="{w - 12}" height="28" rx="14" fill="#FFFFFF" stroke="#667085"/>')
    s.text(x + (w - 12) / 2, 536, label, size=11.5, fill="#344054", weight=600)
    x += w
s.text(x + 4, 536, "gate outbound after reading", size=11, fill="#667085", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q27-prompt-injection-defense.svg")
