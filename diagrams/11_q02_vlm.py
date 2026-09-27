"""Multimodal Q2: how a VLM turns a 336x336 image into 576 prompt tokens."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 440, "Inside a vision-language model", "The image becomes 576 patch tokens that sit in the prompt beside the text; the LLM attends to all of them.")
P = PALETTE["pink"]
# 336x336 image cut into a 24x24 grid of 14px patches (drawn at 6px per patch)
gx, gy, n, c = 40, 100, 24, 6
s.add(f'<rect x="{gx}" y="{gy}" width="{n*c}" height="{n*c}" rx="4" fill="{P[1]}" stroke="{P[0]}" stroke-width="1.6"/>')
for i in range(1, n):
    s.add(f'<line x1="{gx + i*c}" y1="{gy}" x2="{gx + i*c}" y2="{gy + n*c}" stroke="{P[0]}" stroke-opacity="0.35" stroke-width="0.6"/>')
    s.add(f'<line x1="{gx}" y1="{gy + i*c}" x2="{gx + n*c}" y2="{gy + i*c}" stroke="{P[0]}" stroke-opacity="0.35" stroke-width="0.6"/>')
s.add(f'<rect x="{gx + 5*c}" y="{gy + 3*c}" width="{c}" height="{c}" fill="{P[0]}"/>')
s.text(gx + n * c / 2, 262, "336 × 336 image", size=12, weight=700, fill=P[2])
s.text(gx + n * c / 2, 280, "24 × 24 = 576 patches", size=11, fill=P[2])

s.arrow([(188, 172), (214, 172)], "pink")
s.box(216, 132, 190, 80, "ViT encoder", ["CLIP / SigLIP-pretrained", "self-attention over patches"], "pink", size=14)
s.arrow([(406, 172), (438, 172)], "model")
s.box(440, 132, 160, 80, "MLP projector", ["2 layers", "→ LLM hidden size"], "model", size=14)

# the prompt: text tokens, then the image tokens in place of <image>, then more text
C = PALETTE["compute"]
def cells(x0, k):
    for i in range(k):
        s.add(f'<rect x="{x0 + i*23}" y="306" width="20" height="28" rx="4" fill="{C[1]}" stroke="{C[0]}" stroke-width="1.5"/>')
cells(40, 4)
s.add(f'<rect x="136" y="306" width="440" height="28" rx="6" fill="{PALETTE["model"][1]}" stroke="{PALETTE["model"][0]}" stroke-width="1.6"/>')
s.text(356, 320, "576 image tokens, in place of <image>", size=12, weight=600, fill=PALETTE["model"][2])
cells(584, 4)
s.text(40, 356, "text tokens", size=11, fill=C[2], anchor="start")
s.text(676, 356, "text tokens", size=11, fill=C[2], anchor="end")
s.arrow([(520, 212), (520, 304)], "model")

s.arrow([(678, 320), (704, 320)], "model")
s.box(706, 290, 170, 60, "LLM decoder", ["attends to every", "image token"], "model", size=14)
s.arrow([(791, 350), (791, 378)], "output")
s.pill(791, 398, 140, 36, "Text answer", "output", size=12.5)

s.region(630, 84, 246, 176, "Training", "slate", dashed=False)
s.box(646, 112, 214, 60, "1 · Align projector", ["on captions; encoder + LLM frozen"], "amber", size=13, detail=11)
s.box(646, 184, 214, 56, "2 · Instruction-tune", role="output", size=13)

s.text(40, 404, "Downscaling to the encoder's resolution loses small text;", size=11.5, fill="#344054", anchor="start")
s.text(40, 422, "tiling fixes it but multiplies tokens.", size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/11-multimodal-ai/q02-vlm.svg")
