"""Prompt Engineering Q9: prompt-injection defense in layers, and the combination to break."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 412, "Prompt injection: limit what an injected model can do", "The model has no hard line between instructions and data, so the defense is layers around it, not a sentence in the prompt.")

# where the untrusted text comes from
s.pill(125, 112, 200, 36, "Direct: user types it", "human", size=12)
s.pill(125, 172, 210, 52, "Indirect: web page,\nemail, PDF, tool result", "fail", size=11.5)
s.arrow([(225, 112), (240, 112), (254, 136)], "fail")
s.arrow([(230, 172), (240, 172), (254, 148)], "fail")

# row 1: in
s.hexagon(322, 142, 136, 56, "Input classifier", "amber", size=12.5)
s.arrow([(390, 142), (418, 142)], "model")
s.box(420, 108, 170, 68, "LLM", ["untrusted text", "labelled as data"], "model")
s.arrow([(505, 176), (505, 236)], "model")

# row 2: out, right to left
s.hexagon(505, 270, 176, 64, "Output validation\n+ allowlists", "amber", size=12.5)
s.arrow([(417, 270), (392, 270)], "compute")
s.box(230, 238, 160, 64, "Tools", ["least privilege,", "enforced in code"], "compute")
s.arrow([(230, 270), (204, 270)], "human")
s.box(40, 238, 162, 64, "Human confirms", ["irreversible actions"], "human", size=13.5)

s.text(315, 346, "Classifiers and labelled delimiters lower the success rate;", size=11.5, fill="#667085", italic=True)
s.text(315, 366, "they do not eliminate it. Design as if injection will sometimes succeed.", size=11.5, fill="#667085", italic=True)

# the combination
s.region(626, 84, 250, 300, "The dangerous combination", "fail", dashed=False)
for i, (t, role) in enumerate([("Private data", "data"), ("Untrusted content", "fail"), ("A way to send data out", "amber")]):
    y = 120 + i * 78
    s.pill(751, y + 22, 210, 42, t, role, size=12.5)
    if i < 2:
        s.text(751, y + 61, "+", size=18, weight=700, fill="#8E2A23")
s.text(751, 350, "Remove any one of the three.", size=13, weight=700, fill="#275C1C")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/02-prompt-engineering/q09-prompt-injection.svg")
