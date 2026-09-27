"""LLMOps Q48: eliminating single points of failure: a failover ladder behind health checks and breakers."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 540, "No single point of failure", "Every critical dependency gets a pre-tested fallback, tried automatically in order; drills prove it works.")

# entry column
s.pill(124, 110, 170, 40, "Application", "human")
s.arrow([(124, 130), (124, 156)], "human")
s.box(39, 158, 170, 62, "Gateway", ["multi-zone, stateless"], "compute", size=14)
s.arrow([(124, 220), (124, 246)], "compute")
s.hexagon(124, 284, 190, 72, "Health checks\n+ breakers", "amber", size=13)

# failover ladder
s.text(300, 96, "Failover order, lowest risk first", size=12.5, weight=700, fill="#344054", anchor="start")
rungs = [("Primary provider", "normal path", "compute"),
         ("Same model, other cloud or region", "lowest-risk failover", "compute"),
         ("Other provider, own prompt", "prompt variant run in CI evals", "model"),
         ("Self-hosted open model", "", "data"),
         ("Cached or degraded answer", "last resort", "slate")]
BUS = 262
s.add(f'<line x1="{BUS}" y1="136" x2="{BUS}" y2="{136 + 4 * 62}" stroke="{PALETTE["amber"][0]}" stroke-width="2"/>')
s.arrow([(219, 284), (BUS, 284)], "amber", head=False)
for i, (lab, sub, role) in enumerate(rungs):
    y = 136 + i * 62
    s.add(f'<circle cx="{300}" cy="{y}" r="13" fill="{PALETTE[role][0]}"/>')
    s.text(300, y + 1, str(i + 1), size=12, weight=700, fill="#FFFFFF")
    s.arrow([(BUS, y), (285, y)], "amber")
    s.box(322, y - 24, 272, 48, lab, [sub] if sub else [], role, size=12.5, detail=11)

# forgotten SPOFs
st, f, ink = PALETTE["fail"]
s.add(f'<rect x="624" y="86" width="252" height="252" rx="14" fill="{f}" fill-opacity="0.55" stroke="{st}" stroke-opacity="0.6"/>')
s.text(640, 108, "The forgotten SPOFs", size=13, weight=700, fill=ink, anchor="start")
items = [("Query embedder", "self-hosted copy of the same\nmodel; another cannot query the index"),
         ("Vector DB", "replicas, BM25 fallback"),
         ("The gateway itself", "multi-zone"),
         ("Auth, secrets, flags", "cached values, safe defaults")]
y = 134
for head, body in items:
    s.text(640, y, head, size=12, weight=700, fill=ink, anchor="start")
    s.text(640, y + 18 + (6 if "\n" in body else 0), body, size=11, fill="#344054", anchor="start")
    y += 62 if "\n" in body else 50

# the math
s.add('<rect x="24" y="420" width="852" height="98" rx="14" fill="#FFFFFF" stroke="#E4E7EC"/>')
s.text(44, 442, "Idealized availability, two independent providers in parallel", size=12.5, weight=700, fill="#344054", anchor="start")
for i, x in enumerate((44, 200)):
    s.add(f'<rect x="{x}" y="458" width="130" height="36" rx="8" fill="{PALETTE["compute"][1]}" stroke="{PALETTE["compute"][0]}"/>')
    s.text(x + 65, 476, f"provider {i + 1}: 99.5%", size=11.5, weight=600, fill=PALETTE["compute"][2])
s.text(187, 476, "∥", size=16, weight=700, fill="#344054")
s.text(350, 476, "→", size=16, fill="#344054")
s.text(372, 476, "1 − 0.005² = 99.9975%", size=14, weight=700, fill=PALETTE["output"][2], anchor="start", mono=True)
s.text(860, 476, "correlated outages make this an upper bound;\nrun game days that block the primary", size=11.5, fill="#8E2A23", anchor="end")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/08-llmops-and-production-ai/q48-eliminate-spof.svg")
