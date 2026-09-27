"""System design Q25: automated code migration: codemods for the majority, an LLM agent for the long tail, CI on every change."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 454, "Automated code migration", "Deterministic codemods take the mechanical majority; an LLM agent handles the long tail, and CI verifies every change.")
# row 1
s.box(36, 108, 160, 64, "AST inventory", ["30k call sites"], "data", size=13.5)
s.arrow([(196, 140), (222, 140)], "data")
s.box(224, 108, 170, 64, "Dependency order", ["leaves first"], "data", size=13.5)
s.arrow([(394, 140), (420, 140)], "compute")
s.box(422, 108, 160, 64, "Codemods", ["~70%, mechanical"], "compute", size=13.5)
s.arrow([(582, 140), (612, 140)], "compute")
s.hexagon(690, 140, 156, 64, "Compile\n+ tests", "amber")
# agent loop under CI
s.arrow([(668, 172), (668, 238)], "fail")
s.text(658, 205, "unmatched or failing", size=11.5, fill="#8E2A23", weight=600, anchor="end")
s.box(596, 240, 190, 70, "LLM agent", ["guide + examples already", "migrated in this repo"], "model", size=13.5)
s.arrow([(712, 240), (712, 174)], "model")
s.text(722, 205, "retry", size=11.5, fill="#3B2596", weight=600, anchor="start")
s.arrow([(596, 275), (512, 275)], "slate", label="gave up", label_dy=-11)
s.pill(430, 275, 160, 40, "Human queue", "slate")
s.text(691, 326, "the other ~30%: ~9k sites in ~3k files", size=11, fill="#3B2596", anchor="middle")
# pass: differential tests then small PRs
s.arrow([(768, 140), (830, 140), (830, 360)], "output")
s.text(800, 128, "pass", size=11.5, fill="#275C1C", weight=600, anchor="end")
s.hexagon(800, 392, 150, 60, "Differential\ntests", "amber", size=12.5)
s.arrow([(725, 392), (684, 392)], "output")
s.box(440, 356, 242, 72, "Small PRs, by owner", ["owner-reviewed; flag deletions", "and test edits"], "output", size=13.5)
s.text(40, 380, "The constraint is CI capacity", size=11.5, fill="#344054", weight=600, anchor="start")
s.text(40, 398, "and review bandwidth, not tokens.", size=11.5, fill="#344054", weight=600, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q25-code-migration.svg")
