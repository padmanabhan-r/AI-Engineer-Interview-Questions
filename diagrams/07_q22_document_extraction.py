"""System design Q22: document extraction: layout OCR, typed extraction, validation, and human review for the rest."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 460, "Data extraction from unstructured documents", "The model extracts into a schema, but validation and review decide shipped accuracy; every value links to its source box.")
# row 1: ingest to extraction
s.pill(96, 140, 144, 44, "Email, upload,\nscan", "human", size=12.5)
s.arrow([(168, 140), (194, 140)], "compute")
s.box(196, 106, 190, 68, "OCR + layout", ["tables, reading order,", "bounding boxes"], "compute", size=13.5)
s.arrow([(386, 140), (412, 140)], "compute")
s.box(414, 110, 150, 60, "Type classifier", ["30+ document types"], "compute", size=13)
s.arrow([(564, 140), (590, 140)], "model")
s.box(592, 106, 270, 68, "Schema-constrained LLM", ["typed JSON, explicit 'not present',", "source span per field"], "model", size=13.5)
# row 2: validation
s.arrow([(700, 174), (700, 232)], "model")
s.hexagon(700, 264, 210, 62, "Validation", "amber")
s.text(652, 311, "sums, formats, master data", size=11, fill="#7A5300")
s.arrow([(595, 264), (452, 264)], "output", label="pass · straight-through", label_dy=-11)
s.box(250, 234, 200, 62, "Downstream ERP", ["JSON, versioned schema"], "output", size=13.5)
# row 3: human review
s.arrow([(760, 295), (760, 364)], "fail")
s.text(770, 334, "fail or low", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.text(770, 349, "confidence", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.box(520, 366, 260, 64, "Human review UI", ["source box highlighted per field"], "human", size=13.5)
s.arrow([(520, 382), (350, 382), (350, 298)], "human")
s.arrow([(520, 414), (277, 414)], "data")
s.cylinder(190, 406, 170, 62, "Eval + fine-tune data", "data", size=12)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q22-document-extraction.svg")
