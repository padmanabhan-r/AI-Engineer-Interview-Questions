"""System design Q10: an LLM inference platform, gateway and router over vLLM pools, with a metrics-driven control plane."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 530, "An LLM inference platform", "Route by prefix to warm caches, scale on queue wait and KV pressure, and meter every team.")

# request path
s.pill(78, 200, 104, 42, "Teams", "human")
s.arrow([(130, 200), (152, 200)], "human")
s.hexagon(228, 200, 148, 62, "Gateway", "amber", size=13.5)
s.text(228, 246, "auth · quotas", size=11.5, fill="#7A5300", weight=600)
s.text(228, 262, "metering", size=11.5, fill="#7A5300", weight=600)
s.arrow([(302, 200), (326, 200)], "amber")
s.box(328, 160, 164, 80, "Router", ["prefix affinity,", "else least", "outstanding tokens"], "compute", size=14)

s.region(530, 84, 346, 214, "vLLM replica pools", "model", label_pos="bl")
s.arrow([(492, 200), (510, 200), (510, 138), (548, 138)], "compute")
s.arrow([(510, 200), (510, 218), (548, 218)], "compute")
s.box(550, 106, 306, 64, "70B pool", ["continuous batching, paged KV, prefix cache"], "model", size=14)
s.box(550, 186, 306, 64, "8B pool + multi-LoRA", ["many team fine-tunes on one base"], "model", size=14)

# control plane
s.region(24, 356, 852, 150, "CONTROL PLANE", "slate", label_pos="bl")
s.box(660, 394, 196, 76, "Metrics", ["queue wait, KV use,", "TTFT, TPOT"], "slate", size=14)
s.arrow([(758, 298), (758, 392)], "model")
s.arrow([(660, 432), (608, 432)], "slate")
s.box(440, 400, 166, 64, "Autoscaler", ["adds or drains replicas"], "compute", size=14)
s.arrow([(590, 400), (590, 300)], "compute")
s.cylinder(210, 432, 176, 76, "", "slate")
s.text(210, 434, "Registry +\ncanary deployer", size=12.5, weight=600, fill="#344054")
s.arrow([(298, 432), (330, 432), (330, 324), (548, 324), (548, 300)], "slate", dashed=True)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q10-inference-platform.svg")
