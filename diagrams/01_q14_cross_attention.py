"""LLM Fundamentals Q14: cross-attention: queries from the target, keys and values from the source."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 490, "Cross-attention: one stream conditions on another", "Queries come from the target; keys and values come from the source and are computed once.")
# source lane
s.pill(92, 128, 120, 40, "Source", "data")
s.arrow([(152, 128), (178, 128)], "data")
s.box(180, 96, 140, 64, "Encoder", ["reads the source"], "data")
s.arrow([(320, 128), (372, 128)], "data")
s.cylinder(440, 128, 132, 70, "Source K, V", "data", size=12.5)
s.text(520, 120, "computed once and cached;", size=11.5, fill="#0A5A51", anchor="start")
s.text(520, 138, "never grows during generation", size=11.5, fill="#0A5A51", anchor="start")

# decoder block
s.region(180, 212, 560, 128, "Decoder block", "model", label_pos="br")
s.pill(96, 280, 140, 40, "Target so far", "human", size=12.5)
s.arrow([(166, 280), (194, 280)], "human")
s.pill(272, 280, 148, 38, "Self-attention", "model", size=12.5)
s.text(272, 312, "causal", size=11, fill="#3B2596")
s.arrow([(346, 280), (370, 280)], "model")
s.text(358, 266, "Q", size=12, weight=700, fill="#3B2596")
s.box(372, 246, 136, 68, "Cross-attention", ["Q meets K, V"], "compute", size=13)
s.arrow([(440, 163), (440, 244)], "data")
s.text(450, 196, "K, V", size=12, weight=700, fill="#0A5A51", anchor="start")
s.arrow([(508, 280), (534, 280)], "compute")
s.pill(576, 280, 80, 36, "FFN", "amber", size=12.5)
s.arrow([(616, 280), (744, 280)], "amber")
s.pill(806, 280, 120, 40, "Next token", "output", size=12.5)

# score shape
col = PALETTE["compute"][0]
for i in range(4):
    for j in range(7):
        s.add(f'<rect x="{44 + j * 18}" y="{388 + i * 18}" width="16" height="16" rx="3" fill="{col}" fill-opacity="0.75" stroke="{col}"/>')
s.text(184, 392, "scores: T_target × T_source", size=12, weight=700, fill="#1B418C", anchor="start")
s.text(184, 412, "every target token sees every", size=11.5, fill="#344054", anchor="start")
s.text(184, 428, "source token: no causal mask", size=11.5, fill="#344054", anchor="start")
s.text(184, 444, "over the source", size=11.5, fill="#344054", anchor="start")

# where it shows up
s.box(430, 376, 206, 84, "Encoder-decoder", ["T5, Whisper-style: between", "self-attention and the FFN"], "model", size=13)
s.box(650, 376, 206, 84, "Text-to-image diffusion", ["Q: image latents", "K, V: text embeddings"], "pink", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q14-cross-attention.svg")
