"""System design Q43: financial document processing: extraction wrapped in reconciliation, maker-checker review and an immutable audit trail."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 600, "Document processing for a financial institution", "Extraction wrapped in controls: values reconcile or go to a human, and every step is on the audit trail.")

# row 1
s.pill(88, 125, 120, 42, "Documents", "slate")
s.arrow([(148, 125), (170, 125)], "slate")
s.box(172, 94, 176, 62, "Encrypt + classify", ["sensitivity, in-region"], "slate", size=13.5)
s.arrow([(348, 125), (370, 125)], "compute")
s.box(372, 94, 146, 62, "OCR + layout", [], "compute", size=13.5)
s.arrow([(518, 125), (540, 125)], "model")
s.box(542, 94, 310, 62, "Schema-bound extraction", ["every value keeps its source span"], "model", size=13.5)

# row 2: the two checks
s.arrow([(760, 156), (760, 216)], "amber")
s.hexagon(760, 250, 196, 64, "Reconcile\nbalances + totals", "amber", size=12.5)
s.arrow([(620, 156), (620, 190), (470, 190), (470, 219)], "fail")
s.hexagon(470, 250, 180, 62, "Tamper + fraud\nsignals", "fail", size=12.5)

# row 3: decision paths
s.arrow([(760, 282), (760, 368)], "output")
s.text(770, 326, "pass", size=11.5, fill="#275C1C", weight=600, anchor="start")
s.arrow([(662, 250), (618, 250), (618, 392), (572, 392)], "fail")
s.text(628, 326, "fail", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.arrow([(470, 281), (470, 368)], "fail")
s.box(370, 370, 202, 62, "Maker-checker", ["reviewer sees source highlight"], "human", size=13.5)
s.arrow([(572, 416), (668, 416)], "human")
s.box(670, 370, 182, 62, "Core systems", ["rules or humans decide"], "output", size=13.5)
s.arrow([(760, 432), (760, 476)], "data")
s.cylinder(760, 510, 200, 66, "Immutable audit", "data")
s.text(760, 560, "inputs, model versions, reviewers", size=11, fill="#0A5A51", weight=600)
s.text(470, 456, "no autonomous adverse decisions", size=11.5, fill="#1F3864", weight=600)

# the controls, in words
s.region(24, 186, 318, 380, "", "slate", dashed=False)
rows = [
    (236, "Reconciliation beats confidence", 700, "#7A5300", 44),
    (262, "opening balance + transactions", 400, "#344054", 44),
    (282, "= closing balance", 400, "#344054", 60),
    (306, "income matches across pay slips", 400, "#344054", 44),
    (326, "and bank statements", 400, "#344054", 60),
    (372, "Model governance", 700, "#3B2596", 44),
    (398, "pinned versions, documented", 400, "#344054", 44),
    (418, "validation, monitoring", 400, "#344054", 60),
    (442, "an unpinned provider alias", 400, "#344054", 44),
    (462, "fails audit", 400, "#344054", 60),
    (508, "Data residency", 700, "#0A5A51", 44),
    (534, "approved in-region endpoint,", 400, "#344054", 44),
    (554, "PII encrypted", 400, "#344054", 60),
]
for y, t, w, c, x in rows:
    s.text(x, y - 20, t, size=12.5 if w == 700 else 12, weight=w, fill=c, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q43-financial-doc-pipeline.svg")
