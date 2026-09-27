"""System design Q12: a text-to-image service as an async GPU job system with safety gates on both sides."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 520, "A text-to-image generation service", "An async GPU job system: gate the prompt, queue by tier, denoise in latent space, then gate the image itself.")

# request and queueing
s.pill(76, 142, 100, 42, "User", "human")
s.arrow([(126, 142), (148, 142)], "human")
s.box(150, 110, 130, 64, "API", ["credits, tier"], "slate", size=14)
s.arrow([(280, 142), (304, 142)], "slate")
s.hexagon(372, 142, 136, 62, "Prompt\nsafety", "fail", size=12.5)
s.arrow([(440, 142), (456, 142), (456, 116), (474, 116)], "fail")
s.arrow([(456, 142), (456, 168), (474, 168)], "fail")
s.pill(560, 116, 172, 36, "Fast · reserved", "amber", size=12)
s.pill(560, 168, 172, 36, "Relaxed · idle GPUs", "slate", size=12)
s.arrow([(646, 116), (658, 116), (658, 142), (670, 142)], "amber")
s.arrow([(646, 168), (658, 168), (658, 142)], "slate", head=False)
s.box(672, 106, 188, 72, "Batch scheduler", ["same model +", "resolution"], "compute", size=13.5)

# GPU worker, right to left
s.region(24, 222, 852, 128, "GPU WORKER · latent diffusion", "model", label_pos="bl")
s.arrow([(765, 178), (765, 258)], "compute")
s.pill(765, 280, 170, 42, "Text encoder", "model")
s.arrow([(680, 280), (626, 280)], "model")
s.pill(500, 280, 250, 42, "Denoiser · 20–30 steps", "model")
s.arrow([(375, 280), (321, 280)], "model", label="latent", label_dy=-11)
s.pill(236, 280, 170, 42, "VAE decode", "model")
s.text(500, 316, "most of the GPU-seconds", size=11.5, fill="#3B2596", weight=600)

# outputs
s.arrow([(236, 301), (236, 396)], "model")
s.text(246, 376, "pixels", size=11.5, fill="#3B2596", weight=600, anchor="start")
s.hexagon(236, 430, 220, 64, "Output safety\n+ watermark", "fail", size=12.5)
s.arrow([(346, 430), (398, 430)], "fail")
s.cylinder(478, 430, 156, 70, "Storage + CDN", "data", size=12.5)
s.arrow([(600, 301), (600, 430), (668, 430)], "model", dashed=True)
s.pill(764, 430, 190, 42, "Live previews", "output", size=12.5)
s.text(764, 466, "progressive, via websocket", size=11.5, fill="#275C1C", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q12-text-to-image-service.svg")
