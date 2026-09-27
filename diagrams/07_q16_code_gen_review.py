"""System design Q16: code generation and review: two products (IDE completions, PR review) on one context engine."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 600, "Code generation and review on one context engine", "Completions need speed from a small model; review needs precision, so only verified findings get posted.")
# top lane: completions
s.region(24, 80, 852, 110, "IDE · completions under ~300 ms", "compute")
s.pill(120, 142, 150, 40, "IDE, at cursor", "human")
s.arrow([(195, 142), (258, 142)], "compute")
s.box(260, 112, 300, 60, "FIM completions", ["small model · code before + after cursor"], "compute", size=13.5)
s.arrow([(560, 142), (618, 142)], "output")
s.pill(720, 142, 200, 40, "Inline completion", "output")

# middle: shared context engine
s.region(24, 208, 852, 104, "SHARED CONTEXT ENGINE", "data", label_pos="tr")
s.cylinder(120, 262, 160, 76, "Repo + past\nreviews", "data", size=12.5)
s.arrow([(200, 262), (258, 262)], "data")
s.box(260, 232, 300, 60, "Context index", ["symbols, call graph, embeddings"], "data", size=13.5)
s.arrow([(410, 232), (410, 174)], "data")

# bottom lane: PR review
s.region(24, 330, 852, 250, "PR REVIEW · within minutes", "model", label_pos="tr")
s.arrow([(410, 292), (410, 370)], "data")
s.pill(120, 402, 150, 40, "PR webhook", "human")
s.arrow([(195, 402), (258, 402)], "model")
s.box(260, 372, 300, 60, "Review context", ["changed symbols, callers, tests"], "model", size=13.5)
s.arrow([(560, 402), (598, 402)], "model")
s.box(600, 372, 190, 60, "LLM reviewer", ["candidate findings"], "model", size=13.5)
s.arrow([(740, 432), (740, 488)], "amber")
s.hexagon(740, 520, 220, 60, "Verifier", "amber")
s.text(740, 564, "evidence + confidence per finding", size=11, fill="#7A5300")
s.arrow([(630, 520), (562, 520)], "output", label="high only", label_dy=-11)
s.box(300, 490, 260, 62, "Inline comments", ["never approves or merges"], "output", size=13.5)
s.arrow([(300, 521), (222, 521)], "slate")
s.pill(130, 521, 170, 44, "Resolved or\ndismissed", "slate", size=12)
s.arrow([(130, 499), (130, 460), (660, 460), (660, 434)], "slate", dashed=True)
s.text(400, 474, "learn from dismissals: suppress categories", size=11, fill="#344054", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q16-code-gen-review.svg")
