"""LLMOps Q45: switching LLM providers without downtime: port, evaluate, shadow, canary, ramp, keep a fallback."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 540, "Switching LLM providers without downtime", "The API swap is a config change behind a gateway; matching behaviour is the real work, proven offline, then on live traffic.")
W, H, Y1, Y2 = 180, 66, 100, 236
X = [32, 250, 468, 686]
cx = [x + W / 2 for x in X]

s.box(X[0], Y1, W, H, "Gateway abstraction", ["adapters absorb format", "and error differences"], "compute", size=13)
s.box(X[1], Y1, W, H, "Port prompts + tools", ["re-tune per provider"], "model", size=13)
s.hexagon(cx[2], Y1 + H / 2, W, H, "Offline evals", "amber")
s.text(cx[2], Y1 - 12, "regression, safety, latency suites", size=11, fill="#7A5300", weight=600)
s.box(X[3], Y1, W, H, "Shadow traffic", ["compare judge scores,", "length, cost"], "data", size=13)
for i in range(3):
    s.arrow([(X[i] + W, Y1 + H / 2), (X[i + 1] - 2, Y1 + H / 2)], "slate")

# row 2, right to left
s.box(X[3], Y2, W, H, "Canary 1–10%", ["sticky per conversation"], "amber", size=13)
s.arrow([(cx[3] + 40, Y1 + H), (cx[3] + 40, Y2 - 2)], "data")
s.box(X[2], Y2, W, H, "Ramp to 100%", ["automatic rollback", "triggers"], "output", size=13)
s.arrow([(X[3], Y2 + H / 2), (X[2] + W + 2, Y2 + H / 2)], "output")
s.pill(340, Y2 + H / 2, 214, 46, "Old provider = fallback", "slate", size=12.5)
s.arrow([(X[2], Y2 + H / 2), (449, Y2 + H / 2)], "slate")
# regression loop back to porting
s.arrow([(cx[3] - 40, Y2), (cx[3] - 40, 204), (cx[1], 204), (cx[1], Y1 + H + 2)], "fail", dashed=True)
s.text(cx[1] + 12, 190, "regression: back to porting", size=11.5, weight=600, fill="#8E2A23", anchor="start")

# traffic share panel
s.text(32, 346, "Share of live traffic served by the new provider", size=12.5, weight=700, fill="#344054", anchor="start")
bx, bw = 160, 520
rows = [("Shadow", 0, "copy only, answers not served"), ("Canary", 0.06, "1–10%"), ("Ramp", 1.0, "100%, old provider on standby")]
nf, ns = PALETTE["model"][1], PALETTE["model"][0]
of, os_ = PALETTE["slate"][1], PALETTE["slate"][0]
for i, (lab, share, note) in enumerate(rows):
    y = 368 + i * 40
    s.text(bx - 12, y + 12, lab, size=12, weight=600, fill="#344054", anchor="end")
    s.add(f'<rect x="{bx}" y="{y}" width="{bw}" height="24" rx="6" fill="{of}" stroke="{os_}"/>')
    if share > 0:
        s.add(f'<rect x="{bx}" y="{y}" width="{bw * share:.1f}" height="24" rx="6" fill="{nf}" stroke="{ns}"/>')
    else:
        s.add(f'<rect x="{bx + 3}" y="{y + 3}" width="{bw - 6}" height="18" rx="5" fill="none" stroke="{ns}" stroke-dasharray="5 4"/>')
    s.text(bx + bw + 12, y + 12, note, size=11.5, fill="#344054", anchor="start")
s.text(bx + bw / 2, 380, "old provider serves every answer", size=11, fill="#344054")
s.text(bx + bw / 2, 420, "old provider serves the rest", size=11, fill="#344054")
s.text(bx + bw / 2, 460, "new provider serves", size=11, fill=PALETTE["model"][2], weight=600)

s.add('<rect x="32" y="490" width="836" height="34" rx="10" fill="#FFF4D6" stroke="#C98A06" stroke-opacity="0.5"/>')
s.text(450, 507, "Embeddings are not swappable: re-embed the corpus in the background, dual-write, then switch embedder and index together.", size=11.5, fill="#7A5300", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/08-llmops-and-production-ai/q45-provider-switch.svg")
