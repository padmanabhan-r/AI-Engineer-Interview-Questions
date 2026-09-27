"""System design Q36: an organisation-wide AI gateway: a stateless data plane of gates, cache and router, fed by a control plane."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 576, "An AI gateway for the whole organisation", "One entry point: policy, budgets, guardrails, caching and failover built once, adding only milliseconds.")

s.pill(126, 116, 170, 42, "Apps + agents", "human")
s.text(126, 150, "per-app credentials", size=11, fill="#1F3864", weight=600)
s.box(520, 86, 300, 60, "Control plane", ["vault keys · policies · prices"], "slate", size=13.5)
s.arrow([(670, 146), (670, 196)], "slate", dashed=True)
s.text(680, 172, "cached config", size=11.5, fill="#344054", weight=600, anchor="start")

# data plane
s.region(24, 184, 852, 240, "DATA PLANE · stateless, ~10–20 ms p99 overhead", "compute", label_pos="bl")
s.arrow([(126, 160), (126, 222)], "human")
s.hexagon(126, 254, 180, 58, "AuthN + policy", "human", size=13)
s.text(126, 298, "model · data class · region", size=11, fill="#1F3864", weight=600)
s.arrow([(216, 254), (240, 254)], "compute")
s.hexagon(332, 254, 180, 58, "Token limits\n+ budgets", "amber", size=13)
s.arrow([(422, 254), (446, 254)], "compute")
s.hexagon(538, 254, 180, 58, "Guardrails", "fail", size=13)
s.text(538, 298, "PII, secrets, injection", size=11, fill="#8E2A23", weight=600)
s.arrow([(628, 254), (668, 254)], "compute")
s.cylinder(740, 254, 136, 64, "Cache", "data")
s.arrow([(740, 286), (740, 356), (652, 356)], "compute")
s.box(430, 326, 220, 60, "Router", ["fallback, circuit breakers"], "compute", size=13.5)
s.arrow([(430, 356), (330, 356)], "data")
s.cylinder(222, 356, 212, 60, "Metering + redacted logs", "data", size=12.5)

# providers
s.arrow([(540, 386), (540, 466)], "model")
s.arrow([(540, 440), (330, 440), (330, 466)], "model")
s.arrow([(540, 440), (750, 440), (750, 466)], "model")
s.box(250, 468, 160, 56, "Provider A", [], "model", size=13.5)
s.box(460, 468, 160, 56, "Provider B", [], "model", size=13.5)
s.box(670, 468, 160, 56, "Self-hosted pool", [], "model", size=13.5)
s.text(450, 550, "Provider keys never leave the vault; direct egress to provider domains is blocked.", size=11.5, fill="#344054")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q36-ai-gateway.svg")
