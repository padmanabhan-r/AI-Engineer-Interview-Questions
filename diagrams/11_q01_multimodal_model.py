"""Multimodal Q1: per-modality encoders, a projector, and one backbone over an interleaved sequence."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 440, "How a multimodal model reads several inputs", "Each modality becomes vectors in the LLM's embedding space, and one transformer attends over all of them together.")
s.pill(80, 130, 100, 38, "Image", "pink")
s.pill(80, 225, 100, 38, "Audio", "amber")
s.pill(80, 320, 100, 38, "Text", "compute")
s.box(160, 101, 170, 58, "Vision encoder", ["ViT patch embeddings"], "pink", size=13.5)
s.box(160, 196, 170, 58, "Audio encoder", ["log-Mel frames"], "amber", size=13.5)
s.box(160, 291, 170, 58, "Tokenizer", ["text token IDs"], "compute", size=13.5)
for y, r in [(130, "pink"), (225, "amber"), (320, "compute")]:
    s.arrow([(130, y), (158, y)], r)
s.box(380, 140, 160, 70, "Projector", ["MLP, resampler or", "query transformer"], "model", size=13.5)
s.arrow([(330, 130), (378, 158)], "pink")
s.arrow([(330, 225), (378, 195)], "amber")

# the interleaved token sequence the backbone sees
seq = ["compute", "compute", "pink", "pink", "pink", "pink", "amber", "amber", "compute", "compute"]
sx, sy, c = 590, 110, 22
for i, r in enumerate(seq):
    col, fill, _ = PALETTE[r]
    s.add(f'<rect x="{sx}" y="{sy + i * 25}" width="{c}" height="{c}" rx="4" fill="{fill}" stroke="{col}" stroke-width="1.6"/>')
s.text(sx + c / 2, 94, "one sequence", size=11, weight=600, fill="#344054")
s.arrow([(540, 175), (584, 175)], "model")
s.arrow([(330, 320), (584, 320)], "compute")

s.box(640, 110, 220, 110, "Transformer backbone", ["attends over interleaved", "text, image and audio tokens"], "model", size=14)
s.arrow([(616, 165), (638, 165)], "model")
s.arrow([(750, 220), (750, 272)], "output")
s.box(640, 274, 220, 66, "Output", ["text, or tokens for an", "image or audio decoder"], "output", size=14)

s.text(36, 392, "Token-heavy: one image is typically hundreds to a few thousand tokens, and that volume drives latency and cost.",
       size=11.5, fill="#344054", anchor="start")
s.text(36, 414, "CLIP-style dual encoders are different: they embed for search and classification, and do not generate.",
       size=11.5, fill="#667085", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/11-multimodal-ai/q01-multimodal-model.svg")
