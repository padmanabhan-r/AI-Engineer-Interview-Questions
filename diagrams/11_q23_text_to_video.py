"""Multimodal Q23: text-to-video with a video VAE and a diffusion transformer over spacetime patches."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 420, "Text-to-video with a diffusion transformer", "A video VAE compresses the clip; a transformer denoises spacetime patches jointly, which keeps frames consistent.")
s.box(30, 98, 170, 56, "Prompt", ["+ reference image"], "human", size=14)
s.arrow([(200, 126), (248, 126)], "human")
s.box(250, 94, 170, 64, "Encoders", ["T5-class or LLM text", "+ image encoder"], "compute", size=14)

# the noisy latent: a stack of frames, each cut into patches
L = PALETTE["slate"]
for i in range(3, -1, -1):
    x, y = 50 + i * 14, 250 - i * 10
    s.add(f'<rect x="{x}" y="{y}" width="110" height="70" rx="4" fill="{L[1]}" stroke="{L[0]}" stroke-width="1.4"/>')
x, y = 50, 250
for k in range(1, 4):
    s.add(f'<line x1="{x + k*27.5}" y1="{y}" x2="{x + k*27.5}" y2="{y + 70}" stroke="{L[0]}" stroke-opacity="0.5"/>')
for k in (1, 2):
    s.add(f'<line x1="{x}" y1="{y + k*23.3:.1f}" x2="{x + 110}" y2="{y + k*23.3:.1f}" stroke="{L[0]}" stroke-opacity="0.5"/>')
import random
random.seed(7)
for _ in range(40):
    s.add(f'<circle cx="{x + 4 + random.random()*102:.1f}" cy="{y + 4 + random.random()*62:.1f}" r="1.3" fill="{L[0]}" opacity="0.45"/>')
s.add(f'<rect x="{x + 27.5}" y="{y + 23.3}" width="27.5" height="23.3" fill="{PALETTE["model"][0]}" fill-opacity="0.35" stroke="{PALETTE["model"][0]}" stroke-width="1.4"/>')
s.text(126, 346, "noise in a spacetime latent,", size=11.5, weight=600, fill=L[2])
s.text(126, 363, "cut into spacetime patches", size=11, fill=L[2])

s.box(440, 170, 200, 120, "Diffusion transformer", ["denoises all patches jointly", "attention across frames", "diffusion or flow matching"], "model", size=14, detail=11)
s.arrow([(420, 126), (540, 126), (540, 168)], "compute")
s.arrow([(218, 262), (438, 262)], "slate")
s.arrow([(640, 230), (678, 230)], "data")
s.box(680, 195, 170, 70, "Video VAE decoder", ["3D, often causal"], "data", size=13.5)
s.arrow([(765, 265), (765, 300)], "output")
s.pill(765, 320, 170, 36, "Clip, then upscale", "output", size=12)

s.text(30, 398, "Still fails on physics, object permanence and long-range consistency; deepfake risk needs watermarking and C2PA provenance.",
       size=11.5, fill=PALETTE["fail"][2], anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/11-multimodal-ai/q23-text-to-video.svg")
