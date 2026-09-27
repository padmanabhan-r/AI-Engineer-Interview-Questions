"""Prompt Engineering Q22: an output parser between the model's text and deterministic code."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 500, "Output parser: raw text in, typed object out", "It is the boundary between a probabilistic generator and deterministic code: parse, validate, or fail loudly.")
Y = 146
s.pill(84, Y, 116, 40, "Raw text", "model")
s.arrow([(142, Y), (170, Y)], "compute")
s.box(172, Y - 32, 156, 64, "Extract", ["strip fences, preamble"], "compute", size=13.5)
s.arrow([(328, Y), (350, Y)], "compute")
s.box(352, Y - 32, 160, 64, "Parse + coerce", ["dates, numbers, enums"], "compute", size=13.5)
s.arrow([(512, Y), (540, Y)], "compute")
s.diamond(604, Y, 124, 84, "Valid?", "amber", size=13)
s.text(604, Y - 62, "required fields, ranges, business rules", size=11, fill="#7A5300")
s.arrow([(666, Y), (724, Y)], "output", label="yes", label_dy=-10)
s.pill(796, Y, 140, 40, "Typed object", "output")

# recover path
s.arrow([(604, Y + 42), (604, 250)], "fail")
s.text(614, 218, "no", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.box(540, 252, 264, 62, "Recover", ["retry with the error, a cheap repair", "call, or a fallback plus an alert"], "fail", size=13.5)
s.arrow([(540, 283), (84, 283), (84, Y + 22)], "fail", dashed=True)
s.text(310, 272, "retry: send the error back to the model", size=11.5, fill="#8E2A23", weight=600)

# a worked specimen
def panel(x, y, w, h, head, lines, role):
    st, fi, ink = PALETTE[role]
    s.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#FFFFFF" stroke="{st}" stroke-width="1.4"/>')
    s.text(x + 14, y + 18, head, size=11.5, weight=700, fill=ink, anchor="start")
    for i, l in enumerate(lines):
        s.text(x + 14, y + 42 + i * 17, l, size=11, fill=INK_, anchor="start", mono=True)
INK_ = "#344054"
panel(28, 350, 250, 124, "the model wrote", ["Sure! Here is the order:", "```json", '{"order_id": "A-17",', '\u00a0"date": "3 March 2025"}', "```"], "model")
s.arrow([(278, 412), (318, 412)], "compute")
s.text(298, 398, "parse", size=11, fill="#1B418C", weight=600)
panel(320, 350, 250, 124, "your code gets", ["Order(", '\u00a0\u00a0order_id="A-17",', "\u00a0\u00a0date=date(2025, 3, 3))"], "output")

st, fi, ink = PALETTE["fail"]
s.box(600, 350, 276, 56, "Valid syntax is not a true value", ["a well-formed order_id may not exist"], "fail", size=12.5)
s.box(600, 418, 276, 56, "Watch the parse-failure rate", ["per prompt and model version"], "slate", size=12.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/02-prompt-engineering/q22-output-parser.svg")
