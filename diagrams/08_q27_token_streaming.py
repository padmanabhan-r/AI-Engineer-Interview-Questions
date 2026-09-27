"""LLMOps Q27: token streaming over Server-Sent Events, and why it cuts perceived latency."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 560, "Token streaming over SSE", "Each token goes out as soon as it is sampled; generation is no faster, but the user waits only for the first token.")
LX = {"C": 100, "S": 300, "E": 500}
roles = {"C": "human", "S": "compute", "E": "model"}
names = {"C": "Client", "S": "Server", "E": "Engine"}
for k, x in LX.items():
    s.pill(x, 100, 120, 36, names[k], roles[k])
    s.add(f'<line x1="{x}" y1="118" x2="{x}" y2="392" stroke="{PALETTE[roles[k]][0]}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.6"/>')


def msg(a, b, y, label, role, dashed=False):
    xa, xb = LX[a], LX[b]
    d = 1 if xb > xa else -1
    s.arrow([(xa + 4 * d, y), (xb - 6 * d, y)], role, dashed=dashed)
    s.text((xa + xb) / 2, y - 11, label, size=11.5, weight=600, fill=PALETTE[role][2])


msg("C", "S", 150, "POST, stream: true", "human")
msg("S", "E", 184, "enqueue", "compute")
msg("E", "S", 222, "first token, after prefill", "model", dashed=True)
msg("S", "C", 256, "200 event-stream, first delta", "compute", dashed=True)
s.region(40, 272, 520, 82, "", "slate")
s.text(400, 340, "loop: each decode step", size=12, weight=700, fill="#344054")
msg("E", "S", 304, "token", "model", dashed=True)
msg("S", "C", 336, "data: delta", "compute", dashed=True)
msg("S", "C", 380, "final chunk: usage, then DONE", "output", dashed=True)

# what is on the wire
s.add('<rect x="590" y="84" width="286" height="244" rx="12" fill="#1E2330"/>')
s.text(606, 104, "on the wire", size=11.5, weight=700, fill="#98A2B3", anchor="start")
wire = [("HTTP/1.1 200 OK", "#E4E7EC"), ("Content-Type: text/event-stream", "#E4E7EC"), ("", ""),
        ('data: {"delta": "The"}', "#9FE3D8"), ('data: {"delta": " cat"}', "#9FE3D8"), ('data: {"delta": " sat"}', "#9FE3D8"),
        ("  ...", "#98A2B3"), ('data: {"finish_reason": "stop",', "#B7E4A8"), ('\u00a0\u00a0\u00a0\u00a0\u00a0\u00a0"usage": {...}}', "#B7E4A8"), ("data: [DONE]", "#B7E4A8")]
for i, (l, c) in enumerate(wire):
    if l:
        s.text(606, 130 + i * 19, l, size=11, fill=c, anchor="start", mono=True)
s.text(733, 348, "each event: data: {json} + blank line", size=11, fill="#667085")
s.text(733, 372, "An error after the first byte cannot\nchange the status: send an error event.", size=11, fill="#8E2A23", weight=600)

# perceived latency
s.text(40, 432, "What the user waits for", size=12.5, weight=700, fill="#344054", anchor="start")
bx, bw = 200, 640
st, f, _ = PALETTE["slate"]
s.text(190, 462, "no streaming", size=11.5, fill="#344054", anchor="end")
s.add(f'<rect x="{bx}" y="450" width="{bw}" height="24" rx="6" fill="{f}" stroke="{st}"/>')
s.text(bx + bw / 2, 462, "blank screen for the total time", size=11, fill="#344054")
s.add(f'<circle cx="{bx + bw}" cy="462" r="5" fill="{PALETTE["output"][0]}"/>')
s.text(190, 500, "streaming", size=11.5, fill="#344054", anchor="end")
ttft = 96
s.add(f'<rect x="{bx}" y="488" width="{ttft}" height="24" rx="6" fill="{PALETTE["model"][1]}" stroke="{PALETTE["model"][0]}"/>')
s.text(bx + ttft / 2, 500, "prefill", size=11, fill=PALETTE["model"][2])
for i in range(22):
    x = bx + ttft + 6 + i * 24.5
    s.add(f'<rect x="{x:.1f}" y="490" width="18" height="20" rx="4" fill="{PALETTE["output"][1]}" stroke="{PALETTE["output"][0]}"/>')
s.arrow([(bx + ttft, 526), (bx + ttft, 516)], "model")
s.text(bx + ttft + 8, 534, "time to first token: text appears here", size=11, fill=PALETTE["model"][2], weight=600, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/08-llmops-and-production-ai/q27-token-streaming.svg")
