"""System design Q33: graceful degradation as a ladder of rungs, each still useful, chosen per feature."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 520, "The degradation ladder", "Each rung is still useful; health signals step down automatically, and the user is told which rung they are on.")

rungs = [
    ("Primary model", "full features", "model"),
    ("Fallback model", "reduced features", "model"),
    ("Cached answers", "cached + precomputed", "data"),
    ("Retrieval-only", "show top documents; no hidden LLM call", "compute"),
    ("Rules + templates", "direct APIs; manual path stays alive", "slate"),
    ("Honest message", "queue the task, offer a human", "human"),
]
triggers = ["failing", "failing", "cache miss", "search down", "else"]
W, H, DX, DY, X0, Y0 = 270, 56, 104, 70, 40, 84
for i, (t, d, role) in enumerate(rungs):
    x, y = X0 + i * DX, Y0 + i * DY
    s.box(x, y, W, H, t, [d], role, size=13.5)
    if i < len(rungs) - 1:
        ax = x + 52
        s.arrow([(ax, y + H), (ax, y + DY + H / 2), (x + DX - 2, y + DY + H / 2)], "fail" if i < 4 else "slate", width=1.8)
        s.text(ax - 8, y + H + 13, triggers[i], size=11, fill="#8E2A23" if i < 4 else "#344054", weight=600, anchor="end")
    s.text(x + W + 12, y + H / 2, f"L{i}", size=12, weight=700, fill="#98A2B3", anchor="start", mono=True)

# per-feature floor (top right, where the ladder leaves room)
s.region(470, 78, 390, 124, "Pick the floor per feature", "amber", dashed=False)
s.text(490, 118, "Medical answer", size=12.5, weight=700, fill="#7A5300", anchor="start")
s.text(632, 118, "→ straight to “unavailable”", size=12, fill="#344054", anchor="start")
s.text(490, 146, "Autocomplete", size=12.5, weight=700, fill="#7A5300", anchor="start")
s.text(632, 146, "→ vanishes silently", size=12, fill="#344054", anchor="start")
s.text(490, 174, "Summaries, drafts", size=12.5, weight=700, fill="#7A5300", anchor="start")
s.text(632, 174, "→ queued, notify when done", size=12, fill="#344054", anchor="start")

# what moves you down (bottom left, under the ladder)
s.hexagon(140, 396, 180, 46, "Health signals", "fail", size=13)
s.text(140, 430, "step down automatically", size=11.5, fill="#8E2A23", weight=600)
s.hexagon(140, 474, 180, 44, "Feature flag", "amber", size=13)
s.text(244, 474, "on-call forces a rung", size=11.5, fill="#7A5300", weight=600, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q33-degradation-ladder.svg")
