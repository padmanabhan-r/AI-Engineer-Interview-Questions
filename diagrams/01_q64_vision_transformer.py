"""LLM Fundamentals Q64: ViT cuts an image into patches that become tokens; a projector feeds them to an LLM."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 470, "Transformers on images: patches become tokens", "A Transformer processes any sequence of vectors; ViT makes that sequence from 16×16 image patches.")
s.region(490, 124, 376, 260, "LLaVA-style", "model", label_pos="tr")
# the image as a 14 x 14 patch grid
col, fill, ink = PALETTE["data"]
for i in range(14):
    for j in range(14):
        op = 0.35 + 0.5 * (((i * 7 + j * 3) % 11) / 10)
        s.add(f'<rect x="{40 + j * 8}" y="{124 + i * 8}" width="7" height="7" fill="{col}" fill-opacity="{op:.2f}"/>')
s.add(f'<rect x="39" y="123" width="113" height="113" rx="3" fill="none" stroke="{col}" stroke-width="1.6"/>')
s.text(96, 110, "224×224 image", size=12, weight=700, fill=ink)
s.text(96, 252, "14 × 14 = 196 patches", size=11, fill=ink)
s.text(96, 267, "of 16×16 pixels", size=11, fill=ink)
s.arrow([(158, 180), (186, 180)], "data")

# token strip
x = 190
for lab, w, role in [("CLS", 48, "amber"), ("p1", 40, "data"), ("p2", 40, "data"), ("p3", 40, "data"), ("…", 30, "slate"), ("p196", 48, "data")]:
    c, f, k = PALETTE[role]
    s.add(f'<rect x="{x}" y="162" width="{w}" height="36" rx="7" fill="{f}" stroke="{c}" stroke-width="1.5"/>')
    s.text(x + w / 2, 181, lab, size=11.5, weight=700, fill=k, mono=True)
    x += w + 5
s.text(330, 216, "flatten + linear projection,", size=11, fill="#0A5A51")
s.text(330, 231, "+ position embeddings", size=11, fill="#0A5A51")
s.arrow([(467, 180), (500, 180)], "data")
s.box(502, 144, 160, 72, "ViT encoder", ["standard Transformer", "encoder"], "model")
s.arrow([(582, 216), (582, 296)], "model")
s.box(502, 298, 160, 64, "Projector", ["MLP"], "amber")
s.arrow([(662, 330), (698, 330)], "amber")
s.box(700, 290, 156, 80, "Decoder LLM", ["visual tokens join", "its sequence"], "model", size=13.5)
s.pill(778, 186, 130, 40, "Text tokens", "human", size=12.5)
s.arrow([(778, 206), (778, 288)], "human")
s.arrow([(778, 370), (778, 398)], "output")
s.pill(778, 420, 120, 40, "Answer", "output")

# alternatives and limits
s.box(40, 296, 212, 64, "Or: Flamingo-style", ["cross-attention to image", "features instead"], "compute", size=13)
s.box(40, 372, 212, 76, "No built-in locality", ["unlike a CNN, so ViT needed", "much larger pretraining", "data to beat CNNs"], "slate", size=13)
s.box(266, 296, 212, 152, "Still weak at", ["counting", "spatial relations", "small text", "", "tiling helps but multiplies", "tokens and cost"], "fail", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q64-vision-transformer.svg")
