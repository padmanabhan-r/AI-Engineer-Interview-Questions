"""Agents Q30: human-in-the-loop: a durable pause on a risky tool call, with the decision fed back as the tool result."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 556, "Human-in-the-loop: pause, checkpoint, resume", "Policy stops a costly action, the state is saved, and the person's decision comes back as the pending call's result.")

L = {"agent": 100, "harness": 305, "human": 510}
for k, label, role in [("agent", "Agent", "model"), ("harness", "Harness", "amber"), ("human", "Human", "human")]:
    s.box(L[k] - 66, 84, 132, 40, label, (), role, size=13.5)
    s.add(f'<line x1="{L[k]}" y1="124" x2="{L[k]}" y2="440" stroke="{PALETTE[role][0]}" stroke-width="1.5" stroke-dasharray="4 5" opacity="0.6"/>')


def msg(y, a, b, label, role, dashed=False, mono=False, note=""):
    x0, x1 = L[a], L[b]
    d = 1 if x1 > x0 else -1
    s.arrow([(x0 + d * 6, y), (x1 - d * 8, y)], role, dashed=dashed)
    s.text((x0 + x1) / 2, y - 12, label, size=11.5, fill=PALETTE[role][2], mono=mono, weight=600 if not mono else 400)
    if note:
        s.text((x0 + x1) / 2, y + 13, note, size=10.5, fill="#667085")


msg(160, "agent", "harness", "issue_refund(480 USD)", "model", mono=True)
s.hexagon(L["harness"], 208, 196, 40, "policy: over 200 needs approval", "amber", size=11.5)
s.pill(L["harness"], 262, 164, 34, "checkpoint state", "slate", size=12)
s.text(L["harness"] + 92, 262, "process can exit;", size=10.5, fill="#667085", anchor="start")
s.text(L["harness"] + 92, 276, "hours later is fine", size=10.5, fill="#667085", anchor="start")
msg(318, "harness", "human", "approve? exact params + reason", "amber")
msg(364, "human", "harness", "edit to 300 USD", "human", dashed=True, mono=True)
msg(414, "harness", "agent", "resume: decision as tool result", "output")

# tiering
s.region(598, 84, 272, 356, "Tier by reversibility, blast radius", "slate", dashed=False)
s.add('<defs><linearGradient id="risk" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stop-color="#3E8E2F"/>'
      '<stop offset="0.5" stop-color="#C98A06"/><stop offset="1" stop-color="#D0443A"/></linearGradient></defs>')
s.add('<rect x="620" y="126" width="14" height="284" rx="7" fill="url(#risk)"/>')
s.text(648, 140, "irreversible, large", size=12, weight=700, fill="#8E2A23", anchor="start")
s.text(648, 157, "approval always, or disallowed", size=11, fill="#8E2A23", anchor="start")
s.add('<path d="M638,268 L650,261 L650,275 Z" fill="#C98A06"/>')
s.text(656, 260, "refund over 200 USD", size=12, weight=700, fill="#7A5300", anchor="start")
s.text(656, 277, "pause for approval", size=11, fill="#7A5300", anchor="start")
s.text(648, 380, "reversible, small", size=12, weight=700, fill="#275C1C", anchor="start")
s.text(648, 397, "runs autonomously", size=11, fill="#275C1C", anchor="start")
s.text(734, 426, "also: regulated decisions, low confidence", size=10.5, fill="#667085")

s.box(30, 462, 840, 70, "Pitfall: approval fatigue", ["asked 200 times a day, people rubber-stamp", "keep approvals rare, show exact parameters and the reason, track approval and edit rates"], "fail", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q30-human-in-the-loop.svg")
