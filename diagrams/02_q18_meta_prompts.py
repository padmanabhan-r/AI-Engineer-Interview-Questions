"""Prompt Engineering Q18: a meta-prompt inside an evaluation loop becomes prompt optimization."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 500, "Meta-prompts: an LLM writes the prompt, the eval picks it", "Scores and failures feed back into the meta-prompt; a held-out set the optimizer never saw has the last word.")
Y = 150
s.pill(96, Y, 140, 48, "Task +\nexamples", "human", size=12.5)
s.arrow([(166, Y), (194, Y)], "model")
s.box(196, Y - 34, 160, 68, "Meta-prompt", ["drafts, critiques or", "rewrites a prompt"], "model", size=14)
s.arrow([(356, Y), (392, Y)], "model")
st, fi, _ = PALETTE["slate"]
for d in (14, 7):
    s.add(f'<rect x="{394 + d}" y="{Y - 30 - d}" width="150" height="60" rx="12" fill="{fi}" stroke="{st}" stroke-width="1.4" stroke-opacity="0.6"/>')
s.box(394, Y - 30, 150, 60, "Candidate prompts", role="slate", size=13.5)
s.arrow([(558, Y), (600, Y)], "slate")
s.cylinder(662, Y, 120, 78, "Eval set", "data")

# feedback
s.arrow([(662, Y + 39), (662, 256)], "data")
s.box(582, 258, 160, 58, "Scores + failures", role="amber", size=13.5)
s.arrow([(582, 287), (276, 287), (276, Y + 36)], "amber")
s.text(430, 276, "fed back into the meta-prompt", size=11.5, fill="#7A5300", weight=600)
s.arrow([(742, 287), (770, 287), (770, 336)], "output")
s.pill(770, 358, 150, 40, "Best prompt", "output")
s.arrow([(770, 378), (770, 404)], "output")
s.hexagon(770, 436, 176, 58, "Held-out test set\nnever optimized on", "amber", size=12)

# three uses
modes = [("Draft", ["task + ideal outputs", "→ a system prompt"]),
         ("Critique", ["prompt + failing cases", "→ diagnosis + revision"]),
         ("Optimize", ["score candidates,", "feed scores back", "APE · OPRO · DSPy"])]
for i, (t, lines) in enumerate(modes):
    s.box(28 + i * 188, 360, 176, 94, t, lines, "model", size=13.5)
s.text(28, 342, "Three ways to use one", size=12.5, weight=700, fill="#3B2596", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/02-prompt-engineering/q18-meta-prompts.svg")
