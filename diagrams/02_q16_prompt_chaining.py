"""Prompt Engineering Q16: a prompt chain for contract review, with code checks between the LLM steps."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 430, "Prompt chaining: one job per call, code in between", "Each step has a validated output that feeds the next; code checks and branches, and each step gets its own model.")
Y = 166
s.pill(90, Y, 116, 40, "Contract", "human")
s.arrow([(148, Y), (178, Y)], "model")
s.box(180, Y - 32, 160, 64, "1. Extract clauses", ["small model"], "model", size=13.5)
s.arrow([(340, Y), (378, Y)], "model")
s.diamond(440, Y, 124, 88, "Schema\nvalid?", "amber", size=12)
s.arrow([(440, Y - 44), (440, 106), (260, 106), (260, Y - 34)], "fail", dashed=True)
s.text(350, 94, "no: run step 1 again", size=11.5, fill="#8E2A23", weight=600)
s.arrow([(502, Y), (560, Y)], "amber", label="yes", label_dy=-10)

# parallel classify: stacked cards
st, fi, _ = PALETTE["model"]
for d in (16, 8):
    s.add(f'<rect x="{562 + d}" y="{Y - 32 - d}" width="200" height="64" rx="12" fill="{fi}" stroke="{st}" stroke-width="1.4" stroke-opacity="0.6"/>')
s.box(562, Y - 32, 200, 64, "2. Classify clauses", ["in parallel"], "model", size=13.5)

s.arrow([(662, Y + 32), (662, 256)], "model")
s.diamond(662, 300, 130, 88, "High\nrisk?", "amber", size=12)
s.arrow([(727, 300), (766, 300)], "amber", label="no", label_dy=-10)
s.pill(820, 300, 104, 40, "Short\nsummary", "output", size=12)
s.arrow([(597, 300), (542, 300)], "amber", label="yes", label_dy=-10)
s.box(370, 268, 170, 64, "3. Risk analysis", ["strong model"], "model", size=13.5)
s.arrow([(370, 300), (296, 300)], "model")
s.pill(210, 300, 170, 40, "4. Memo for legal", "output", size=12.5)

s.text(28, 380, "Violet = an LLM call.  Amber = code: validate, branch, look up, compute.", size=12, fill="#344054", weight=600, anchor="start")
s.text(28, 402, "Trade-off: errors propagate and latency adds up, so evaluate each step as well as the whole chain.", size=11.5, fill="#667085", italic=True, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/02-prompt-engineering/q16-prompt-chaining.svg")
