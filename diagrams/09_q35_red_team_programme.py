"""Evaluation Q35: pre-launch red teaming as a time-boxed programme with exit criteria agreed in advance."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 500, "Red teaming a chatbot before launch", "A time-boxed programme: scope and exit criteria first, attack, fix and verify until the criteria are met.")
Y = 180

s.box(32, Y - 44, 158, 88, "Scope", ["what it can access", "and do; rubric;", "exit criteria"], "human", size=14)
s.arrow([(190, Y), (214, Y)], "human")
s.pill(282, Y, 132, 42, "Manual", "pink")
s.text(282, Y + 36, "explore first", size=11, fill="#8A1F58", weight=600)
s.arrow([(348, Y), (372, Y)], "pink")
s.pill(440, Y, 132, 42, "Automated", "model")
s.text(440, Y + 36, "attacks at scale", size=11, fill="#3B2596", weight=600)
s.arrow([(506, Y), (530, Y)], "model")
s.pill(578, Y, 92, 42, "Fix", "compute")
s.arrow([(624, Y), (650, Y)], "compute")
s.diamond(730, Y, 156, 100, "Verify: exit\ncriteria met?", "amber")
# not met: back to fix
s.arrow([(730, Y - 50), (730, 102), (578, 102), (578, Y - 23)], "fail")
s.text(654, 90, "not met", size=11.5, weight=600, fill="#8E2A23")
# met: sign-off
s.arrow([(730, Y + 50), (730, 276)], "output")
s.text(740, 256, "met", size=11.5, weight=600, fill="#275C1C", anchor="start")
s.pill(730, 298, 170, 42, "Sign-off, then CI", "output", size=13)

# exit criteria
st, f, ink = PALETTE["amber"]
s.add(f'<rect x="32" y="262" width="300" height="130" rx="14" fill="{f}" stroke="{st}" stroke-width="1.6"/>')
s.text(48, 282, "Exit criteria, agreed in advance", size=12.5, weight=700, fill=ink, anchor="start")
sev = [("S1", "fail"), ("S2", "amber"), ("S3", "compute"), ("S4", "slate")]
for i, (lab, role) in enumerate(sev):
    s.pill(70 + i * 62, 310, 50, 24, lab, role, size=11.5)
s.text(48, 340, "· zero open S1", size=11.5, fill="#344054", anchor="start")
s.text(48, 358, "· S2 below an agreed rate", size=11.5, fill="#344054", anchor="start")
s.text(48, 376, "· no over-refusal regression", size=11.5, fill="#344054", anchor="start")

# fix order
s.text(366, 282, "Fix order", size=12.5, weight=700, fill="#344054", anchor="start")
for i, (lab, role) in enumerate([("1  permissions + architecture", "compute"), ("2  filters", "model"), ("3  prompts, last", "slate")]):
    w = 230 - i * 40
    s.add(f'<rect x="366" y="{296 + i * 32}" width="{w}" height="26" rx="8" fill="{PALETTE[role][1]}" stroke="{PALETTE[role][0]}"/>')
    s.text(378, 309 + i * 32, lab, size=11.5, weight=600, fill=PALETTE[role][2], anchor="start")

s.text(32, 424, "Team: security, domain experts, multilingual testers and non-builders; external red teamers for high exposure.", size=11.5, fill="#344054", anchor="start")
s.add('<rect x="32" y="444" width="836" height="38" rx="10" fill="#FDE8E6" stroke="#D0443A" stroke-opacity="0.5"/>')
s.text(450, 463, "Pitfall: one afternoon of jailbreaks on the bare model. The serious findings live in retrieval and tools.", size=11.5, weight=600, fill="#8E2A23")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/09-evaluation-and-testing/q35-red-team-programme.svg")
