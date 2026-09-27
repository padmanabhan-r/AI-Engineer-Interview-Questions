"""Agents Q22: graph engineering: nodes do the work, code-owned edges route on typed state, cycles are capped."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 530, "Graph engineering: a support-ticket graph", "The model thinks inside nodes; code reads the state and picks every edge, caps every cycle, and checkpoints after each node.")

s.pill(84, 250, 96, 40, "Ticket", "human")
s.arrow([(132, 250), (160, 250)], "human")
s.box(162, 214, 150, 72, "Classify", ["LLM sets category"], "model")
s.text(237, 304, "code reads it, picks the edge", size=11, fill="#3B2596", weight=600)

# conditional edges
s.add(f'<path d="M312,250 H334" fill="none" stroke="{PALETTE["slate"][0]}" stroke-width="2"/>')
s.arrow([(334, 250), (334, 136), (406, 136)], "slate")
s.arrow([(334, 250), (406, 250)], "slate")
s.arrow([(334, 250), (334, 392), (406, 392)], "slate")
s.text(372, 124, "billing", size=11.5, fill="#344054", weight=600)
s.text(372, 238, "technical", size=11.5, fill="#344054", weight=600)
s.text(372, 380, "legal", size=11.5, fill="#344054", weight=600)

s.box(408, 110, 140, 52, "Billing node", (), "compute", size=13.5)
s.box(408, 220, 140, 60, "Tech node", (), "compute", size=13.5)
s.box(408, 364, 140, 56, "Human review", (), "human", size=13.5)

# grounded check
s.arrow([(548, 136), (664, 136), (664, 164)], "compute")
s.arrow([(548, 234), (580, 234), (580, 192), (597, 192)], "compute")
s.hexagon(664, 192, 134, 54, "Grounded check", "amber", size=12.5)
s.arrow([(731, 192), (760, 192)], "output")
s.text(745, 180, "pass", size=11.5, fill="#275C1C", weight=600)
s.pill(819, 192, 114, 40, "Send reply", "output", size=12.5)

# retries: capped cycle back to tech, then human
s.arrow([(640, 219), (640, 266), (550, 266)], "amber", dashed=True)
s.text(596, 282, "fail, retries < 2", size=11, fill="#7A5300", weight=600)
s.arrow([(690, 219), (690, 392), (550, 392)], "fail")
s.text(700, 330, "fail twice", size=11.5, fill="#8E2A23", weight=600, anchor="start")

# typed state + checkpoints
s.region(30, 444, 840, 66, "", "slate", dashed=False)
s.text(48, 477, "Typed state", size=12.5, weight=700, fill="#344054", anchor="start")
chips = [("ticket", "human"), ("category", "model"), ("draft", "compute"), ("retries", "amber")]
x = 150
for name, role in chips:
    col, fill, ink = PALETTE[role]
    s.add(f'<rect x="{x}" y="464" width="88" height="26" rx="13" fill="{fill}" stroke="{col}"/>')
    s.text(x + 44, 477, name, size=11.5, fill=ink, mono=True)
    x += 98
s.text(560, 477, "checkpoint after each node:", size=11.5, fill="#344054", weight=700, anchor="start")
s.text(560, 494, "pause for approval, resume, replay", size=11, fill="#667085", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q22-graph-engineering.svg")
