"""LLM Fundamentals Q41: attention sinks: keep the first 4 tokens plus a rolling window; evict the middle."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 420, "Attention sinks: keep the first tokens, roll the rest", "Softmax must put its mass somewhere; heads with nothing to attend to park it on the first token.")
X0, STEP, C = 44, 42, 36
def cell(i, role, label="", dashed=False, y=132, h=36, faded=False):
    col, fill, ink = PALETTE[role]
    x = X0 + i * STEP
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    op = ' opacity="0.55"' if faded else ""
    s.add(f'<rect x="{x}" y="{y}" width="{C}" height="{h}" rx="7" fill="{fill}" stroke="{col}" stroke-width="1.6"{dash}{op}/>')
    if label:
        s.text(x + C / 2, y + h / 2 + 1, label, size=12, weight=700, fill=ink, mono=True, opacity=0.6 if faded else 1)

for i in range(16):
    if i < 4: cell(i, "model", str(i))
    elif i < 10: cell(i, "fail", "×", dashed=True, faded=True)
    else: cell(i, "compute")
s.pill(812, 150, 76, 36, "query", "amber", size=12.5)
s.arrow([(794, 132), (700, 84), (200, 84), (122, 128)], "amber", curve=True)
s.arrow([(774, 150), (716, 150)], "amber")
s.text(470, 116, "the current query still attends to the sinks", size=11.5, weight=600, fill="#7A5300")

s.text(125, 186, "sinks 0–3", size=12, weight=700, fill="#3B2596")
s.text(125, 201, "kept forever", size=11, fill="#3B2596")
s.text(314, 186, "middle tokens", size=12, weight=700, fill="#8E2A23")
s.text(314, 201, "evicted, cannot be recalled", size=11, fill="#8E2A23")
s.text(587, 186, "rolling window", size=12, weight=700, fill="#1B418C")
s.text(587, 201, "last W tokens", size=11, fill="#1B418C")

# positions by cache slot
s.text(X0, 230, "position RoPE sees = cache slot, so distances never exceed training", size=11.5, weight=600, fill="#344054", anchor="start")
for i in range(4): cell(i, "slate", str(i), y=244, h=26)
for k, i in enumerate(range(10, 16)): cell(i, "slate", str(4 + k), y=244, h=26)

s.box(44, 300, 262, 88, "Why a sink forms", ["weights must sum to 1; under causal", "masking token 0 is visible to every", "query, so it becomes the no-op"], "model", size=13)
s.box(320, 300, 262, 88, "Evict the sink", ["the parked mass spills onto real", "tokens and perplexity explodes", "(StreamingLLM, 2023)"], "fail", size=13)
s.box(596, 300, 260, 88, "Built in now", ["a learnable sink token, or a", "learned per-head sink logit", "in the softmax"], "output", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q41-attention-sinks.svg")
