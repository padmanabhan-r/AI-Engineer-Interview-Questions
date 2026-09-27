"""Evaluation Q4: evaluation-driven development: evals first, ship only when no slice regresses."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 470, "Evaluation-driven development", "Test-driven development for non-deterministic systems: write the evals first, ship a change only if no slice regresses.")
W, H, Y1, Y2 = 150, 66, 104, 262
X = [32, 212, 392, 572]
DX = 700  # decision centre
cx = [x + W / 2 for x in X]

s.box(X[0], Y1, W, H, "Define criteria", ["binary beats 1–10"], "human", size=13)
s.box(X[1], Y1, W, H, "Build eval set", ["30–50 real cases", "from error analysis"], "amber", size=13)
s.box(X[2], Y1, W, H, "Baseline", ["score today's", "system"], "slate", size=13)
s.box(X[3], Y1, W, H, "Change system", ["prompt, model,", "retrieval or tool"], "model", size=13)
for i in range(3):
    s.arrow([(X[i] + W, Y1 + H / 2), (X[i + 1] - 2, Y1 + H / 2)], "slate")

# decision under "change system"
s.arrow([(DX, Y1 + H), (DX, Y2 - 44)], "model")
s.diamond(DX, Y2, 170, 84, "Re-run: any\nslice worse?", "amber")
# regression loops back up on the right
s.arrow([(DX + 85, Y2), (850, Y2), (850, Y1 + H / 2), (X[3] + W + 2, Y1 + H / 2)], "fail")
s.text(760, 196, "yes:\nregression", size=11.5, weight=600, fill="#8E2A23", anchor="start")
# better: ship
s.arrow([(DX - 85, Y2), (X[2] + W + 2, Y2)], "output", label="no: better", label_dy=-11)
s.box(X[2], Y2 - 33, W, H, "Ship behind flag", ["CI shows Δ per slice"], "output", size=13)
s.arrow([(X[2], Y2), (X[1] + W + 2, Y2)], "fail")
s.box(X[1], Y2 - 33, W, H, "Production failures", ["each one becomes", "a new case"], "fail", size=12)
s.arrow([(cx[1], Y2 - 33), (cx[1], Y1 + H + 2)], "amber", width=2.4)

# the PR check, illustrated
st, f, ink = PALETTE["slate"]
s.add(f'<rect x="32" y="232" width="164" height="118" rx="12" fill="#FFFFFF" stroke="#D0D5DD"/>')
s.text(46, 250, "Pull request check", size=12, weight=700, fill="#344054", anchor="start")
for i, (sl, mark, role) in enumerate([("slice A", "▲", "output"), ("slice B", "▲", "output"), ("slice C", "▼", "fail")]):
    y = 274 + i * 22
    s.text(46, y, sl, size=11.5, fill="#344054", anchor="start")
    s.text(108, y, mark + (" better" if mark == "▲" else " worse"), size=11.5, weight=700, fill=PALETTE[role][2], anchor="start")
s.text(46, 338, "one ▼ blocks the ship", size=11, weight=700, fill="#8E2A23", anchor="start")

s.add('<rect x="32" y="388" width="836" height="58" rx="12" fill="#FFF4D6" stroke="#C98A06" stroke-opacity="0.5"/>')
s.text(48, 406, "Pitfall: hundreds of prompt edits against the same 50 cases overfit them.", size=12, weight=700, fill="#7A5300", anchor="start")
s.text(48, 428, "Keep a held-out split and refresh the set from production.", size=11.5, fill="#7A5300", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/09-evaluation-and-testing/q04-eval-driven-development.svg")
