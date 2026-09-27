"""LLM Fundamentals Q75: Recursive Self-Improvement: generate, verify, train, repeat; the verifier is the bottleneck."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 450, "Recursive self-improvement: a loop that improves itself", "I. J. Good's 1965 “intelligence explosion”: each round makes the next one easier.")
s.box(40, 100, 150, 64, "Model n", ["current system"], "model")
s.arrow([(190, 132), (298, 132)], "model")
s.box(300, 100, 180, 64, "Generate", ["data or code"], "compute")
s.arrow([(390, 164), (390, 228)], "compute")
s.hexagon(390, 260, 180, 60, "Verify + filter", "amber")
s.text(390, 304, "the bottleneck", size=12, weight=700, fill="#8E2A23")
s.arrow([(300, 260), (287, 260)], "amber")
s.pill(245, 260, 80, 36, "Train", "compute", size=12.5)
s.arrow([(205, 260), (192, 260)], "compute")
s.box(40, 228, 150, 64, "Model n+1", ["the next system"], "output")
s.arrow([(80, 228), (80, 166)], "output")
s.text(90, 197, "repeat", size=11.5, weight=600, fill="#275C1C", anchor="start")

s.region(512, 84, 344, 226, "Partial forms today", "slate", dashed=False)
s.box(528, 116, 312, 54, "Self-training on its own reasoning", ["generate, filter, retrain (STaR-style)"], "model", size=12.5)
s.box(528, 180, 312, 54, "Models feeding their successors", ["synthetic data and grading"], "compute", size=12.5)
s.box(528, 244, 312, 54, "AI doing AI engineering", ["coding agents; AlphaEvolve reported to", "optimize its own training infrastructure"], "amber", size=12.5)

s.region(24, 328, 832, 106, "Limits", "fail")
s.pill(170, 378, 220, 32, "verifier reliability", "fail", size=12)
s.pill(462, 378, 330, 32, "model collapse from unfiltered self-training", "fail", size=11.5)
s.pill(730, 378, 160, 32, "compute", "fail", size=12)
s.text(440, 414, "where “better” is not checkable, the loop amplifies the evaluator's blind spots; fast loops leave fewer human checkpoints", size=11, fill="#8E2A23")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q75-rsi.svg")
