"""Agents Q5: the ReAct loop, with the Minecraft trace from the answer."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 460, "ReAct: reason, act, observe, repeat", "The model writes a thought and a tool call; the harness runs the tool and feeds the real result back.")
s.pill(140, 108, 170, 40, "Goal / question", "human")
s.arrow([(140, 128), (140, 158)], "human")
s.box(60, 160, 160, 66, "Thought", ["reasons about the", "next step"], "model")
s.arrow([(220, 193), (292, 193)], "model", label="picks a tool", label_dy=-11)
s.hexagon(370, 193, 150, 62, "Action", "amber")
s.text(370, 237, "tool call", size=11, fill="#7A5300", mono=True)
s.arrow([(370, 250), (370, 298)], "amber")
s.text(380, 274, "harness runs it", size=11.5, fill="#7A5300", weight=600, anchor="start")
s.box(290, 300, 160, 62, "Observation", ["real result, fed back"], "data")
s.arrow([(290, 331), (200, 331), (180, 300), (180, 228)], "data", curve=True)
s.text(370, 380, "grounds the next thought", size=11.5, fill="#0A5A51", weight=600)
s.arrow([(100, 226), (100, 376)], "output")
s.text(92, 300, "when done", size=11.5, fill="#275C1C", weight=600, anchor="end")
s.pill(140, 400, 170, 40, "finish → answer", "output", size=12.5)

# the trace, colour-coded by step
s.region(480, 84, 396, 350, "Trace", "slate", dashed=False)
rows = [("model", "Thought:", "who made Minecraft, then who bought them"),
        ("amber", "Action:", 'search["Minecraft developer"]'),
        ("data", "Observation:", "developed by Mojang Studios"),
        ("model", "Thought:", "now find who acquired Mojang"),
        ("amber", "Action:", 'search["Mojang acquisition"]'),
        ("data", "Observation:", "Microsoft acquired Mojang, 2014"),
        ("output", "Action:", 'finish["Microsoft, in 2014"]')]
for i, (role, k, v) in enumerate(rows):
    y = 128 + i * 43
    col, fill, ink = PALETTE[role]
    s.add(f'<rect x="498" y="{y - 16}" width="360" height="34" rx="8" fill="{fill}" stroke="{col}" stroke-opacity="0.5"/>')
    s.text(510, y + 1, k, size=12, weight=700, fill=ink, anchor="start")
    s.text(604, y + 1, v, size=11.5, fill=ink, anchor="start", mono=role != "model")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q05-react.svg")
