"""Safety Q3: input and output guardrails around the LLM call."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 440, "Input and output guardrails", "Independent checks on both sides of the LLM call, each with its own threshold, action and metric.")
Y = 160
AMB = PALETTE["amber"][2]
s.pill(85, Y, 120, 40, "User input", "human")
s.arrow([(145, Y), (178, Y)], "human")
s.hexagon(265, Y, 170, 64, "Input rails", "amber")
s.text(265, 114, "limits · PII · injection · scope", size=11, fill=AMB)
s.arrow([(350, Y), (388, Y)], "compute")
s.box(390, 130, 130, 60, "LLM call", role="model")
s.arrow([(520, Y), (553, Y)], "compute")
s.hexagon(640, Y, 170, 64, "Output rails", "amber")
s.text(640, 114, "schema · harm · grounding · PII", size=11, fill=AMB)
s.arrow([(725, Y), (753, Y)], "output")
s.pill(810, Y, 110, 40, "Response", "output")

s.arrow([(265, 192), (265, 247)], "fail")
s.text(275, 220, "block", size=11.5, weight=600, fill=PALETTE["fail"][2], anchor="start")
s.pill(265, 268, 190, 38, "Safe refusal + log", "fail", size=12.5)
s.arrow([(455, 190), (455, 243)], "model")
s.text(463, 216, "tool call", size=11, fill=PALETTE["model"][2], anchor="start")
s.hexagon(455, 268, 150, 46, "Arg allowlist", "amber", size=12)
s.arrow([(640, 192), (640, 247)], "fail")
s.text(650, 220, "fail", size=11.5, weight=600, fill=PALETTE["fail"][2], anchor="start")
s.pill(640, 268, 200, 38, "Regenerate or abstain", "amber", size=12.5)

s.region(24, 312, 832, 110, "Per-rail design choices", "slate", dashed=False)
s.box(44, 344, 256, 62, "Fail-closed", ["medical safety classifier times out", "→ block"], "fail", size=13.5)
s.box(312, 344, 256, 62, "Fail-open", ["tone check times out", "→ let the output through"], "output", size=13.5)
s.box(580, 344, 256, 62, "Latency", ["input rails run in parallel with the", "call, and cancel it on a block"], "compute", size=13.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/10-ai-safety-ethics-and-responsible-ai/q03-guardrails.svg")
