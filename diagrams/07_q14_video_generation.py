"""System design Q14: a video generation service, a queued multi-GPU DiT job with cheap previews and frame-level safety."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 520, "A video generation service", "Text-to-image with every cost multiplied by frames: preview cheaply, generate small, then upscale and check the frames.")

# request and queue
s.pill(98, 138, 156, 42, "Prompt (+ image)", "human", size=12)
s.arrow([(176, 138), (196, 138)], "human")
s.hexagon(290, 138, 186, 64, "Prompt + likeness\nmoderation", "fail", size=12.5)
s.arrow([(383, 138), (408, 138)], "fail")
s.box(410, 104, 170, 68, "Priority queue", ["credits; paid tiers", "can pre-empt"], "slate", size=13.5)
s.arrow([(580, 138), (636, 138)], "amber")
s.box(638, 104, 222, 68, "Preview", ["low res, few steps"], "amber", size=14)
s.text(749, 190, "kills bad prompts early", size=11.5, fill="#7A5300", weight=600)

# cascade
s.region(24, 222, 852, 128, "CASCADE · minutes of multi-GPU time", "model", label_pos="bl")
s.arrow([(495, 172), (495, 204), (165, 204), (165, 244)], "slate")
s.box(50, 246, 230, 72, "Full job: DiT", ["sequence-parallel, sharded", "spacetime latent patches"], "model", size=14)
s.arrow([(280, 282), (318, 282)], "model")
s.box(320, 250, 176, 64, "Video VAE", ["decode"], "model", size=14)
s.arrow([(496, 282), (534, 282)], "model")
s.box(536, 250, 324, 64, "Spatial + temporal upscaler", ["more pixels, higher frame rate"], "compute", size=13.5)

# safety and provenance, right to left
s.arrow([(698, 314), (698, 400)], "compute")
s.hexagon(698, 434, 220, 64, "Sampled-frame\nmoderation", "fail", size=12.5)
s.arrow([(588, 434), (546, 434)], "fail")
s.box(356, 402, 188, 64, "Watermark + C2PA", ["provenance"], "output", size=13.5)

# sizing
s.text(44, 410, "5 GPU-min per clip", size=12, fill="#344054", weight=700, anchor="start")
s.text(44, 430, "× 200k clips/day", size=12, fill="#344054", anchor="start")
s.text(44, 450, "≈ 700 GPUs flat out", size=12, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q14-video-generation.svg")
