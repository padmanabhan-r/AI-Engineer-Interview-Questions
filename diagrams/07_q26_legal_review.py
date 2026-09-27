"""System design Q26: legal document review: clause-level comparison against a playbook, verified quotes, a lawyer decides."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 500, "AI-powered legal document review", "Clauses are compared with explicit playbook positions; every flag carries a verified quote, and a lawyer decides.")
s.cylinder(605, 112, 210, 52, "Playbook per jurisdiction", "amber", size=12)
s.arrow([(605, 138), (605, 166)], "amber")
# row 1
s.pill(82, 200, 110, 40, "Contracts", "human", size=12.5)
s.arrow([(137, 200), (156, 200)], "data")
s.box(158, 166, 180, 68, "Segment clauses", ["resolve definitions,", "cross-references"], "data", size=13.5)
s.arrow([(338, 200), (362, 200)], "compute")
s.box(364, 170, 140, 60, "Clause", ["classifier"], "compute", size=13.5)
s.arrow([(504, 200), (518, 200)], "model")
s.box(520, 168, 170, 64, "Compare vs", ["playbook positions"], "model", size=13.5)
s.arrow([(690, 200), (714, 200)], "model")
s.box(716, 168, 150, 64, "Flags", ["+ quoted spans, page"], "model", size=13.5)
# row 2
s.arrow([(796, 232), (796, 290)], "amber")
s.hexagon(796, 322, 136, 62, "Quote found\nin source?", "amber", size=12.5)
s.text(796, 366, "not found: dropped", size=11, fill="#8E2A23", weight=600)
s.arrow([(728, 322), (682, 322)], "output", label="yes", label_dy=-11)
s.box(520, 292, 160, 62, "Redlines", ["from approved fallbacks"], "output", size=13.5)
s.arrow([(434, 230), (434, 290)], "compute")
s.box(344, 292, 160, 62, "Diligence table", ["~40 fields per contract"], "compute", size=13)
# row 3: the lawyer
s.arrow([(600, 354), (600, 404)], "output")
s.arrow([(424, 354), (424, 404)], "compute")
s.box(344, 406, 336, 64, "Lawyer reviews and decides", ["every finding cites exact text and page"], "human", size=14)
s.text(40, 410, "zero-retention model access", size=11.5, fill="#344054", weight=600, anchor="start")
s.text(40, 428, "matter-level ethical walls", size=11.5, fill="#344054", weight=600, anchor="start")
s.text(40, 446, "audit trail", size=11.5, fill="#344054", weight=600, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q26-legal-review.svg")
