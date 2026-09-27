"""Safety Q37: an audit trail that can explain any past decision without rerunning it."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 430, "Audit trail for AI decisions", "Store the reason and every version at decision time, so any past decision can be explained without rerunning it.")
M = PALETTE["model"]
s.pill(120, 130, 160, 40, "Decision made", "compute")
s.cylinder(120, 235, 170, 80, "Registries", "model")
s.text(120, 258, "model · prompt · data", size=11, fill=M[2])
s.pill(120, 360, 190, 40, "Human review actions", "human", size=12)

# the decision record, as a card of fields
x0, y0, w, h = 280, 92, 220, 212
s.add(f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="12" fill="#FFFFFF" stroke="{PALETTE["compute"][0]}" stroke-width="1.8"/>')
s.add(f'<rect x="{x0}" y="{y0}" width="{w}" height="34" rx="12" fill="{PALETTE["compute"][1]}"/>')
s.add(f'<rect x="{x0}" y="{y0 + 22}" width="{w}" height="12" fill="{PALETTE["compute"][1]}"/>')
s.text(x0 + w / 2, y0 + 18, "Decision record", size=14, weight=700, fill=PALETTE["compute"][2])
for i, f in enumerate(["request + outcome", "model hash", "prompt version", "RAG snapshot or retrieved text", "reason, stored at decision time"]):
    y = y0 + 44 + i * 32
    s.add(f'<rect x="{x0 + 12}" y="{y}" width="{w - 24}" height="26" rx="6" fill="#F6F8FC" stroke="#D0D5DD"/>')
    s.text(x0 + 24, y + 13, f, size=11.5, fill="#344054", anchor="start")

s.arrow([(200, 130), (278, 130)], "compute")
s.arrow([(205, 235), (278, 235)], "model", dashed=True)
s.text(241, 214, "version ids", size=11, weight=600, fill=M[2])
s.text(241, 250, "by reference", size=11, fill=M[2])

s.box(520, 330, 124, 60, "Durable queue", role="slate", size=13)
s.arrow([(215, 360), (518, 360)], "human")
s.arrow([(500, 214), (582, 214), (582, 328)], "compute")
s.cylinder(755, 360, 150, 80, "WORM store", "data")
s.text(755, 383, "append-only", size=11, fill=PALETTE["data"][2])
s.arrow([(644, 360), (678, 360)], "data")
s.hexagon(680, 240, 126, 48, "Hash chain", "amber", size=12.5)
s.box(770, 214, 106, 52, "Index", ["user + date"], "data", size=13, detail=11)
s.arrow([(728, 322), (694, 266)], "amber")
s.arrow([(790, 322), (814, 268)], "data")

s.region(530, 84, 346, 112, "Governance", "slate", dashed=False)
for i, line in enumerate(["Retention: ≥ 6 months for high-risk (EU AI Act),", "often years for credit",
                          "Drill: explain random past decisions within an hour", "Release gate: decision records emitted and verified"]):
    s.text(548, 124 + i * 18, line, size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/10-ai-safety-ethics-and-responsible-ai/q37-audit-trail.svg")
