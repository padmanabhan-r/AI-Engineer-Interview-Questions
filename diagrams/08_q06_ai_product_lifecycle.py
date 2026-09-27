"""LLMOps Q6: the AI product lifecycle, a lead-in plus a loop that feeds failures back into the eval set."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 420, "The AI product lifecycle is a loop", "Build the eval set before the prototype; every production failure becomes a new eval case.")
W, H, Y1, Y2 = 148, 74, 116, 262
X = [32, 204, 376, 548, 720]
cx = [x + W / 2 for x in X]

s.region(360, 90, 520, 280, "the loop", "amber", label_pos="br")

# row 1, left to right
s.box(X[0], Y1, W, H, "Problem + metric", ["who acts on it,", "which metric moves"], "human", size=13)
s.box(X[1], Y1, W, H, "Baseline", ["rules, search or", "a classical model"], "slate", size=13)
s.box(X[2], Y1, W, H, "Eval set", ["100–500 real cases,", "bar agreed first"], "amber", size=13)
s.box(X[3], Y1, W, H, "Prototype", ["prompt, then RAG,", "then tools, agent"], "model", size=13)
s.box(X[4], Y1, W, H, "Iterate on evals", ["one change", "at a time"], "model", size=13)
for i in range(4):
    s.arrow([(X[i] + W, Y1 + H / 2), (X[i + 1] - 2, Y1 + H / 2)], "slate" if i < 2 else "model")

# row 2, right to left
s.box(X[4], Y2, W, H, "Harden", ["guardrails, PII,", "cost, security"], "fail", size=13)
s.box(X[3], Y2, W, H, "Roll out", ["shadow, canary, A/B", "behind a kill switch"], "compute", size=13)
s.box(X[2], Y2, W, H, "Operate", ["monitor in", "production"], "output", size=13)
s.arrow([(cx[4], Y1 + H), (cx[4], Y2 - 2)], "fail")
s.arrow([(X[4], Y2 + H / 2), (X[3] + W + 2, Y2 + H / 2)], "compute")
s.arrow([(X[3], Y2 + H / 2), (X[2] + W + 2, Y2 + H / 2)], "output")

# feedback: failures become eval cases
s.arrow([(cx[2], Y2), (cx[2], Y1 + H + 2)], "amber", width=2.4)
s.text(cx[2] + 10, (Y1 + H + Y2) / 2, "failures become\neval cases", size=11.5, weight=600, fill="#7A5300", anchor="start")

# the step teams skip
s.text(cx[2], Y1 - 12, "the step teams skip", size=11, weight=700, fill="#7A5300")

# pitfall note in the free corner
s.add('<rect x="32" y="262" width="300" height="74" rx="12" fill="#FDE8E6" stroke="#D0443A" stroke-opacity="0.5"/>')
s.text(48, 284, "Pitfall: the demo-to-production gap", size=12, weight=700, fill="#8E2A23", anchor="start")
s.text(48, 312, "Without an eval set, every prompt fix\nbreaks another case silently.", size=11.5, fill="#8E2A23", anchor="start")

s.text(450, 388, "Simplest architecture first  ·  if a regex solves 90%, stop", size=11.5, fill="#667085")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/08-llmops-and-production-ai/q06-ai-product-lifecycle.svg")
