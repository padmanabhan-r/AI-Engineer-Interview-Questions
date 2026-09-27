"""System design Q11: an LLM evaluation platform, one scorer library for CI gates and production traces."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 500, "An LLM evaluation platform", "One scorer library serves offline CI gates and online traces; human review feeds the datasets and calibrates the judges.")

# inputs
s.box(40, 92, 170, 70, "Candidate config", ["model, prompt,", "retrieval, tools"], "compute", size=13.5)
s.cylinder(125, 262, 170, 92, "", "data")
s.text(125, 258, "Datasets", size=13, fill="#0A5A51", weight=600)
s.text(125, 278, "versioned + sliced", size=11.5, fill="#0A5A51")
s.arrow([(210, 127), (315, 127), (315, 202)], "compute")
s.arrow([(210, 250), (248, 250)], "data")
s.box(250, 204, 130, 70, "Runner", ["cached by", "config × example"], "slate", size=14)

# scoring
s.cylinder(510, 118, 170, 66, "Prod traces", "slate", size=12.5)
s.text(520, 177, "sampled", size=11.5, fill="#344054", weight=600, anchor="start")
s.arrow([(510, 151), (510, 202)], "slate")
s.arrow([(380, 239), (418, 239)], "slate")
s.box(420, 204, 180, 70, "Scorers", ["exact, programmatic,", "LLM judge"], "model", size=14)
s.arrow([(600, 239), (650, 239)], "model")
s.box(652, 204, 210, 70, "Paired comparison", ["vs baseline, bootstrap CI", "per slice"], "compute", size=13.5)
s.arrow([(757, 204), (757, 158)], "amber")
s.hexagon(757, 124, 210, 64, "CI gate +\ndashboards", "amber", size=13)

# human loop
s.arrow([(600, 262), (626, 262), (626, 370), (650, 370)], "human")
s.box(652, 346, 210, 64, "Human review queue", [], "human", size=13.5)
s.arrow([(652, 392), (600, 392)], "human")
s.box(420, 358, 180, 64, "Judge calibration", ["kappa vs human labels"], "amber", size=13.5)
s.arrow([(510, 358), (510, 276)], "amber")
s.arrow([(757, 410), (757, 472), (125, 472), (125, 310)], "human", dashed=True)
s.text(440, 462, "new labelled examples; every incident becomes a regression case", size=11.5, fill="#1F3864", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q11-eval-platform.svg")
