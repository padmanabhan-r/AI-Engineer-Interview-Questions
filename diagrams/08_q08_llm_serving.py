"""LLMOps Q8: serving LLMs in production: a gateway in front of managed APIs or self-hosted engine replicas."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 520, "Serving LLMs in production", "Start on managed APIs; self-host on an inference engine when volume, residency or latency control pays.")
MY = 262  # main row

s.pill(84, MY, 104, 40, "Clients", "human")
s.arrow([(136, MY), (164, MY)], "human")
s.box(166, MY - 32, 150, 64, "LLM gateway", ["one entry point"], "slate", size=13.5)

# managed lane
s.region(350, 84, 526, 88, "MANAGED", "data")
s.box(520, 102, 200, 54, "Managed APIs", ["start here"], "data", size=13.5)
s.arrow([(241, MY - 32), (241, 129), (518, 129)], "data")

# self-hosted lane
s.region(350, 188, 526, 256, "SELF-HOSTED", "compute")
s.arrow([(316, MY), (380, MY)], "compute")
s.box(382, MY - 34, 196, 68, "KV-cache-aware router", ["conversation → replica", "holding its prefix"], "compute", size=13)
E1, E2 = 240, 340  # engine centres
for ey in (E1, E2):
    s.box(642, ey - 38, 218, 76, "Engine replica", ["continuous batching,", "paged KV, prefix cache"], "model", size=13)
s.arrow([(578, MY - 12), (606, MY - 12), (606, E1), (640, E1)], "compute")
s.arrow([(578, MY + 12), (606, MY + 12), (606, E2), (640, E2)], "compute")
s.box(382, 350, 196, 56, "Autoscaler", ["on queue depth or KV use"], "amber", size=13)
s.arrow([(480, 350), (480, MY + 36)], "amber")
s.text(751, 404, "engines: vLLM, SGLang, TensorRT-LLM", size=11.5, fill="#3B2596", weight=600)

# memory panel: weights + KV cache on two 80 GB GPUs
s.text(36, 346, "GPU memory = weights + KV cache", size=12.5, weight=700, fill="#344054", anchor="start")
gx, gy, gw, gh = 36, 364, 136, 30
for i in range(2):
    x = gx + i * (gw + 12)
    wpx = gw * 70 / 80
    s.add(f'<rect x="{x}" y="{gy}" width="{gw}" height="{gh}" rx="6" fill="#FFFFFF" stroke="#98A2B3"/>')
    s.add(f'<rect x="{x}" y="{gy}" width="{wpx:.1f}" height="{gh}" rx="6" fill="{PALETTE["model"][1]}" stroke="{PALETTE["model"][0]}"/>')
    s.add(f'<rect x="{x + wpx:.1f}" y="{gy}" width="{gw - wpx:.1f}" height="{gh}" fill="{PALETTE["data"][0]}" fill-opacity="0.7"/>')
    s.text(x + wpx / 2, gy + gh / 2, "weights", size=11, weight=600, fill=PALETTE["model"][2])
    s.text(x + gw / 2, gy + gh + 14, f"GPU {i + 1} · 80 GB", size=11, fill="#667085")
s.text(36, 432, "70B in FP16 ≈ 140 GB → two GPUs, tensor parallel", size=11.5, fill="#344054", anchor="start")
s.text(36, 452, "teal = KV cache: the leftover sets concurrency", size=11.5, fill="#0A5A51", weight=600, anchor="start")

s.text(613, 470, "Pitfall: benchmarking one request. Load-test p99 under", size=11.5, fill="#8E2A23", anchor="middle")
s.text(613, 488, "concurrency with production-shaped prompt lengths.", size=11.5, fill="#8E2A23", anchor="middle")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/08-llmops-and-production-ai/q08-llm-serving.svg")
