"""LLMOps Q49: handling orchestration failure in a multi-LLM pipeline: validated steps, fallbacks, durable checkpoints."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 540, "A multi-LLM pipeline that survives a broken step", "Validate every step's output, retry or fall back, and checkpoint durably so a failure resumes from the failed step.")
Y = 190

s.add(f'<circle cx="44" cy="{Y}" r="9" fill="#344054"/>')
s.arrow([(53, Y), (76, Y)], "slate")
s.pill(134, Y, 112, 42, "Extract", "model")
s.arrow([(190, Y), (228, Y)], "model")
s.hexagon(302, Y, 144, 60, "Validate", "amber")
s.text(302, Y + 44, "schema + semantic", size=11, fill="#7A5300", weight=600)
# retry loop above
s.arrow([(302, Y - 30), (302, 116), (134, 116), (134, Y - 23)], "amber")
s.text(218, 104, "invalid: retry with the error", size=11.5, weight=600, fill="#7A5300")
# valid path
s.arrow([(374, Y), (456, Y)], "output", label="valid", label_dy=-10)
s.box(458, Y - 34, 150, 68, "Enrich", ["optional: skip + flag", "if it fails"], "compute", size=14)
s.arrow([(608, Y), (643, Y)], "compute")
s.pill(704, Y, 120, 42, "Summarize", "model", size=13)
s.arrow([(764, Y), (790, Y)], "output")
s.pill(830, Y, 76, 40, "Done", "output")
# fallback extract
s.arrow([(302, Y + 56), (302, 290)], "fail")
s.text(312, 272, "retries exhausted", size=11.5, weight=600, fill="#8E2A23", anchor="start")
s.box(222, 292, 160, 66, "Fallback extract", ["other model or", "deterministic method"], "slate", size=13)
s.arrow([(382, 325), (532, 325), (532, Y + 36)], "slate")
# dead letter
s.arrow([(704, Y + 21), (704, 290)], "fail")
s.text(714, 262, "critical failure", size=11.5, weight=600, fill="#8E2A23", anchor="start")
s.cylinder(704, 326, 150, 70, "Dead-letter queue", "fail", size=12.5)
s.text(704, 378, "→ human review", size=11, weight=600, fill="#8E2A23")

# durability rail
s.add(f'<rect x="32" y="404" width="836" height="30" rx="15" fill="{PALETTE["data"][1]}" stroke="{PALETTE["data"][0]}"/>')
s.text(450, 419, "Durable workflow engine: state checkpointed after each step, so completed LLM calls are not re-paid", size=11.5, weight=600, fill=PALETTE["data"][2])

# reliability multiplies
s.text(32, 466, "Reliability multiplies", size=12.5, weight=700, fill="#344054", anchor="start")
for i in range(5):
    x = 32 + i * 92
    s.add(f'<rect x="{x}" y="480" width="80" height="34" rx="8" fill="{PALETTE["compute"][1]}" stroke="{PALETTE["compute"][0]}"/>')
    s.text(x + 40, 497, "98%", size=12.5, weight=700, fill=PALETTE["compute"][2])
    if i < 4:
        s.text(x + 86, 497, "×", size=13, fill="#344054")
s.text(496, 497, "≈ 90% end to end", size=14, weight=700, fill=PALETTE["fail"][2], anchor="start")
s.text(868, 497, "fewer, stronger steps;\nparallelize independent ones", size=11.5, fill="#667085", anchor="end")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/08-llmops-and-production-ai/q49-orchestration-failure.svg")
