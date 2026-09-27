"""LLMOps Q13: guardrails as checks outside the model, on inputs, tool calls and outputs."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 540, "Guardrails sit outside the model", "Prompt instructions are requests; enforcement happens in checks around the model, and in what it is allowed to do.")
Y = 200  # main row

s.pill(76, Y, 96, 40, "Input", "human")
s.arrow([(124, Y), (150, Y)], "human")
s.hexagon(232, Y, 160, 58, "Input guards", "amber")
s.arrow([(312, Y), (354, Y)], "model")
s.box(356, Y - 36, 150, 72, "LLM", ["no enforcement", "of its own"], "model", size=15)
s.arrow([(506, Y), (548, Y)], "amber")
s.hexagon(630, Y, 164, 58, "Output guards", "amber")
s.arrow([(712, Y), (746, Y)], "output")
s.pill(810, Y, 120, 40, "Response", "output")

# tool guards loop under the LLM
s.hexagon(431, 316, 164, 54, "Tool guards", "amber")
s.arrow([(410, Y + 36), (410, 287)], "model")
s.text(400, 263, "tool call", size=11.5, weight=600, fill="#3B2596", anchor="end")
s.arrow([(452, 289), (452, Y + 38)], "amber")
s.text(462, 263, "validated", size=11.5, weight=600, fill="#7A5300", anchor="start")
s.text(528, 316, "The strongest guardrail limits what the\nsystem can do: permissions, allowlists,\nhuman approval for irreversible actions.", size=11.5, fill="#7A5300", anchor="start")

# block paths to a safe reply
s.pill(431, 100, 150, 38, "Safe reply", "fail")
s.arrow([(232, Y - 29), (232, 100), (354, 100)], "fail", label="block", label_at=0.2, label_dy=0)
s.arrow([(630, Y - 29), (630, 100), (508, 100)], "fail", label="fail", label_at=0.2, label_dy=0)

# what each guard checks
cards = [(137, "Input", ["size limits, PII redaction", "injection + jailbreak classifiers", "topic scope"]),
         (336, "Tools", ["per-role allowlists", "argument validation", "least privilege, approval"]),
         (535, "Output", ["schema validation", "moderation, groundedness", "leaked secrets or PII"])]
for x, head, lines in cards:
    st, f, ink = PALETTE["amber"]
    s.add(f'<rect x="{x}" y="372" width="190" height="80" rx="10" fill="#FFFFFF" stroke="{st}" stroke-opacity="0.6"/>')
    s.text(x + 12, 388, head, size=12, weight=700, fill=ink, anchor="start")
    for i, l in enumerate(lines):
        s.text(x + 12, 408 + i * 16, l, size=11, fill="#344054", anchor="start")

# order by cost
s.text(40, 480, "Order by cost:", size=12, weight=700, fill="#344054", anchor="start")
chips = [("regex + schemas", "slate", 142), ("small classifiers (ms)", "compute", 184), ("LLM judge last", "model", 128)]
x = 140
for i, (lab, role, w) in enumerate(chips):
    s.pill(x + w / 2, 480, w, 28, lab, role, size=11.5)
    if i < 2:
        s.arrow([(x + w + 4, 480), (x + w + 22, 480)], "slate")
    x += w + 26
s.text(40, 510, "Action per rule: block, redact, retry with feedback, or escalate; log every trigger and measure false positives too.", size=11.5, fill="#667085", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/08-llmops-and-production-ai/q13-guardrails.svg")
