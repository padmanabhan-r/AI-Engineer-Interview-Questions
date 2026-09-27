"""Evaluation Q3: agent observability: one trace, a tree of spans, with an online eval linking a bad answer to its cause."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 470, "An agent trace explains a bad answer", "One trace per request, one span per step; an online eval flags the answer and the tree shows why.")

TX0, TX1 = 330, 640  # timeline area
s.text(40, 98, "span", size=11.5, weight=700, fill="#667085", anchor="start")
s.text(TX0, 98, "time →", size=11.5, weight=700, fill="#667085", anchor="start")
rows = [  # indent, label, detail, role, start, end
    (0, "Session: refund request", "root span", "human", 0.0, 1.0),
    (1, "LLM: plan", "model + prompt version, tokens", "model", 0.02, 0.24),
    (1, "Tool: lookup_order", "ok", "data", 0.26, 0.42),
    (1, "Tool: issue_refund", "error 403", "fail", 0.44, 0.58),
    (1, "LLM: final answer", "claims the refund succeeded", "model", 0.60, 0.98),
]
for i, (ind, lab, det, role, a, b) in enumerate(rows):
    y = 130 + i * 52
    st, f, ink = PALETTE[role]
    x = 40 + ind * 22
    if ind:
        s.add(f'<path d="M{x - 12},{y - 30} V{y} H{x - 3}" fill="none" stroke="#98A2B3" stroke-width="1.5"/>')
    s.text(x, y - 7, lab, size=12.5, weight=700, fill=ink, anchor="start", mono=lab.startswith("Tool"))
    s.text(x, y + 11, det, size=11, fill=PALETTE["fail"][2] if role == "fail" else "#344054", anchor="start",
           mono=False)
    bx, bw = TX0 + a * (TX1 - TX0), (b - a) * (TX1 - TX0)
    s.add(f'<rect x="{bx:.1f}" y="{y - 12}" width="{bw:.1f}" height="24" rx="6" fill="{f}" stroke="{st}" stroke-width="1.5"/>')
    tag = {"ok": "ok", "error 403": "403"}.get(det)
    if tag:
        s.text(bx + bw / 2, y, tag, size=11, weight=700, fill=ink, mono=True)
s.add(f'<line x1="{TX0}" y1="108" x2="{TX0}" y2="370" stroke="#D0D5DD"/>')

# the eval on the final answer
s.hexagon(770, 338, 176, 56, "Faithfulness: FAIL", "fail", size=12.5)
s.text(770, 378, "online eval score", size=11, fill="#8E2A23", weight=600)
s.arrow([(640, 338), (680, 338)], "fail")
# link back to the cause
s.arrow([(770, 310), (770, 286), (TX0 + 0.58 * (TX1 - TX0) + 6, 286)], "fail", dashed=True)
s.text(775, 258, "cause: the 403 the\nanswer ignored", size=11.5, weight=600, fill="#8E2A23")

# what each span carries
s.add('<rect x="40" y="404" width="820" height="46" rx="10" fill="#FFFFFF" stroke="#E4E7EC"/>')
s.text(56, 418, "LLM spans:", size=11.5, weight=700, fill="#3B2596", anchor="start")
s.text(130, 418, "model and prompt version, token counts, finish reason, latency", size=11.5, fill="#344054", anchor="start")
s.text(56, 437, "Tool spans:", size=11.5, weight=700, fill="#0A5A51", anchor="start")
s.text(130, 437, "arguments, results or document IDs, errors, duration  ·  trace IDs propagate across services", size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/09-evaluation-and-testing/q03-agent-trace.svg")
