"""Evaluation Q23: evaluating multi-turn conversations with a simulated user and a judge that scores the whole transcript."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 500, "Evaluating a whole conversation", "A simulated user plays it out live, since a replayed transcript diverges after the first different answer.")
LX = {"S": 110, "A": 320, "J": 516}
roles = {"S": "human", "A": "model", "J": "amber"}
names = {"S": "Simulated user", "A": "Assistant", "J": "Judge"}
for k, x in LX.items():
    s.pill(x, 104, 150, 38, names[k], roles[k], size=13)
    y0 = 150 if k == "S" else 123
    s.add(f'<line x1="{x}" y1="{y0}" x2="{x}" y2="398" stroke="{PALETTE[roles[k]][0]}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.6"/>')
s.text(LX["S"], 138, "persona + hidden goal", size=11, fill="#1F3864", weight=600)


def msg(a, b, y, label, role, dashed=False):
    xa, xb = LX[a], LX[b]
    d = 1 if xb > xa else -1
    s.arrow([(xa + 4 * d, y), (xb - 6 * d, y)], role, dashed=dashed)
    s.text((xa + xb) / 2, y - 11, label, size=11.5, weight=600, fill=PALETTE[role][2])


msg("S", "A", 178, "turn 1: request", "human")
msg("A", "S", 216, "response", "model", dashed=True)
msg("S", "A", 262, "turn 2: correction or detail", "human")
msg("A", "S", 300, "response", "model", dashed=True)
s.text((LX["S"] + LX["A"]) / 2, 340, "⋮  more turns", size=12, weight=700, fill="#667085")
msg("A", "J", 380, "transcript + final state", "model")
s.text(LX["J"], 414, "scores the whole run", size=11, fill="#7A5300", weight=600)

# what the judge scores
st, f, ink = PALETTE["amber"]
s.add(f'<rect x="612" y="84" width="264" height="206" rx="14" fill="{f}" stroke="{st}" stroke-width="1.6"/>')
s.text(628, 106, "Judge scores the conversation", size=12.5, weight=700, fill=ink, anchor="start")
for i, l in enumerate(["goal completion", "turns to resolution", "context retention", "consistency", "correction handling"]):
    y = 134 + i * 30
    s.add(f'<rect x="628" y="{y - 12}" width="232" height="24" rx="12" fill="#FFFFFF" stroke="{st}" stroke-opacity="0.5"/>')
    s.text(744, y, l, size=11.5, weight=600, fill=ink)

# probes
s.add('<rect x="612" y="306" width="264" height="114" rx="14" fill="#FFFFFF" stroke="#D0D5DD"/>')
s.text(628, 326, "Scripted probes", size=12.5, weight=700, fill="#344054", anchor="start")
s.text(628, 348, "retention: turn 8 depends on turn 2", size=11.5, fill="#344054", anchor="start")
s.text(628, 368, "corrections and ambiguity", size=11.5, fill="#344054", anchor="start")
s.text(628, 388, "multi-turn escalation (safety)", size=11.5, fill="#344054", anchor="start")
s.text(628, 408, "each scenario run several times", size=11.5, fill="#667085", anchor="start")

s.add('<rect x="32" y="440" width="844" height="40" rx="12" fill="#FDE8E6" stroke="#D0443A" stroke-opacity="0.5"/>')
s.text(454, 460, "Pitfall: simulators are too cooperative. Add terse, vague and adversarial personas; review real transcripts weekly.", size=11.5, weight=600, fill="#8E2A23")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/09-evaluation-and-testing/q23-multi-turn-eval.svg")
