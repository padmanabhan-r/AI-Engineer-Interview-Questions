"""System design Q45: resume screening that redacts, matches each stated requirement with evidence, and leaves every rejection to a human."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 516, "Resume screening at 100K applications a week", "Match stated requirements with quoted evidence, rank and explain; a recruiter decides, and adverse impact is monitored.")

# row 1
s.pill(92, 122, 132, 42, "Application", "slate")
s.arrow([(158, 122), (178, 122)], "compute")
s.box(180, 92, 188, 60, "Parse", ["resume → structured profile"], "compute", size=13.5)
s.arrow([(368, 122), (398, 122)], "pink")
s.hexagon(500, 122, 204, 58, "Redact identifiers", "pink", size=13)
s.text(500, 166, "names, photos, dates of birth, addresses", size=11, fill="#8A1F58", weight=600)
s.arrow([(602, 122), (750, 122), (750, 214)], "pink")

# row 2
s.box(640, 216, 222, 70, "Match per requirement", ["quoted evidence for each"], "model", size=13.5)
s.cylinder(470, 251, 196, 66, "Job requirements\nmust / nice-to-have", "data", size=12)
s.arrow([(568, 251), (638, 251)], "data")
s.text(470, 300, "agreed with the hiring manager", size=11, fill="#0A5A51", weight=600)

col, fill, ink = PALETTE["model"]
s.add(f'<rect x="36" y="206" width="300" height="94" rx="10" fill="{fill}" stroke="{col}" stroke-opacity="0.5"/>')
s.text(52, 226, "Evidence, not a fit score", size=12.5, weight=700, fill=ink, anchor="start")
s.text(52, 252, "5 years Python", size=12, weight=700, fill="#344054", anchor="start")
s.text(166, 252, "→ yes: roles X and Y", size=12, fill="#344054", anchor="start")
s.text(52, 280, "explainable to recruiters and candidates", size=11.5, fill=ink, anchor="start")

# row 3, right to left
s.arrow([(750, 286), (750, 388)], "compute")
s.box(640, 390, 222, 62, "Rank + explanation", ["prioritises, never rejects"], "compute", size=13.5)
s.arrow([(640, 421), (594, 421)], "human")
s.box(372, 390, 220, 62, "Recruiter decides", ["humans make every rejection"], "human", size=13.5)
s.arrow([(372, 421), (294, 421)], "amber")
s.hexagon(176, 421, 234, 62, "Adverse impact\nmonitoring", "amber", size=13)
s.text(176, 470, "group selection rate < 0.8 × highest", size=11, fill="#7A5300", weight=600)
s.text(176, 488, "(four-fifths rule) → investigate", size=11, fill="#7A5300")
s.arrow([(176, 390), (176, 346), (690, 346), (690, 288)], "amber", dashed=True)
s.text(330, 334, "audits the matching", size=11.5, fill="#7A5300", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q45-resume-screening.svg")
