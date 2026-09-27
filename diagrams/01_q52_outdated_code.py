"""LLM Fundamentals Q52: fix deprecated-API code with versioned context plus a lint, type-check and test loop."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 420, "Outdated code: give current truth, reject stale output", "Version-matched docs go into context, and the environment rejects deprecated code until it is clean.")
s.cylinder(92, 148, 112, 70, "Lockfile", "data")
s.text(92, 200, "real versions:", size=11, fill="#0A5A51")
s.text(92, 215, "“X 2.x, not the 1.x API”", size=11, fill="#0A5A51")
s.arrow([(148, 148), (174, 148)], "data")
s.box(176, 110, 192, 76, "Retrieve docs", ["versioned docs, changelogs,", "migration guides"], "data", size=13.5)
s.arrow([(368, 148), (394, 148)], "data")
s.box(396, 110, 160, 76, "Generate code", ["+ in-repo code already", "on the new API"], "model", size=13.5)
s.arrow([(556, 148), (580, 148)], "model")
s.hexagon(662, 148, 160, 62, "Lint · types · tests", "amber", size=12.5)
s.arrow([(742, 148), (764, 148)], "output", label="clean", label_dy=-14)
s.pill(812, 148, 92, 38, "Return", "output", size=12.5)
s.arrow([(662, 179), (662, 236), (476, 236), (476, 190)], "fail")
s.text(574, 252, "errors or deprecation warnings, fed back as errors", size=11.5, weight=600, fill="#8E2A23")

s.box(24, 290, 266, 96, "Why it happens", ["training cutoff, and old APIs", "outnumber new ones in the", "training data"], "fail", size=13)
s.box(307, 290, 266, 96, "Not fine-tuning", ["newer code baked into the", "weights goes stale again"], "slate", size=13)
s.box(590, 290, 266, 96, "Blind spot", ["the loop catches only what tests", "and linters cover: keep an eval", "of recently changed APIs"], "pink", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q52-outdated-code.svg")
