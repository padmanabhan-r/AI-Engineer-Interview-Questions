"""Evaluation Q11: red teaming an LLM application: threat model, manual then automated attacks, triage, fix, regress."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 460, "Red teaming an LLM application", "Attack the whole system as an adversary would, then turn every successful attack into a regression test.")
W, H, Y1, Y2 = 164, 70, 110, 262

# threat model card
st, f, ink = PALETTE["human"]
s.add(f'<rect x="32" y="{Y1 - 16}" width="216" height="236" rx="14" fill="{f}" stroke="{st}" stroke-width="1.8"/>')
s.add(f'<rect x="32" y="{Y1 - 16}" width="6" height="236" rx="3" fill="{st}"/>')
s.text(140, Y1 + 6, "Threat model + policy", size=13.5, weight=700, fill=ink)
threats = [("jailbreaks", "pink"), ("injection via documents", "fail"), ("system-prompt or data\nexfiltration", "fail"),
           ("tool misuse", "amber"), ("cost abuse", "slate")]
y = Y1 + 36
for lab, role in threats:
    h = 38 if "\n" in lab else 26
    s.pill(140, y + h / 2, 192, h, lab, role, size=11.5, weight=600)
    y += h + 7
s.text(140, Y1 + 236, "harm policy with severity levels", size=11, fill=ink, weight=600)

# the loop
X = [282, 480, 678]
s.arrow([(248, Y1 + H / 2), (X[0] - 2, Y1 + H / 2)], "human")
s.box(X[0], Y1, W, H, "Manual attacks", ["security, domain,", "multilingual testers"], "pink", size=13)
s.arrow([(X[0] + W, Y1 + H / 2), (X[1] - 2, Y1 + H / 2)], "pink")
s.box(X[1], Y1, W, H, "Automated attacks", ["garak, PyRIT,", "attacker LLMs"], "model", size=13)
s.arrow([(X[1] + W, Y1 + H / 2), (X[2] - 2, Y1 + H / 2)], "model")
s.box(X[2], Y1, W, H, "Triage", ["by severity"], "amber", size=13)
s.arrow([(X[2] + W / 2, Y1 + H), (X[2] + W / 2, Y2 - 2)], "amber")
s.box(X[2], Y2, W, H, "Fix", ["at the right layer"], "compute", size=13)
s.arrow([(X[2], Y2 + H / 2), (X[1] + W / 2 + (W + 10) / 2 + 3, Y2 + H / 2)], "compute")
s.hexagon(X[1] + W / 2, Y2 + H / 2, W + 10, H, "Regression suite", "output", size=13)
s.arrow([(X[1] + W / 2, Y2), (X[1] + W / 2, Y1 + H + 2)], "output", width=2.4)
s.text(X[1] + W / 2 + 10, (Y1 + H + Y2) / 2, "re-run every\nsuccessful attack", size=11.5, weight=600, fill="#275C1C", anchor="start")

s.add('<rect x="282" y="368" width="586" height="64" rx="12" fill="#FFF4D6" stroke="#C98A06" stroke-opacity="0.5"/>')
s.text(298, 388, "The worst findings come from what the system can do, not what it can say.", size=12, weight=700, fill="#7A5300", anchor="start")
s.text(298, 410, "Spend the budget on tool permissions and indirect injection.", size=11.5, fill="#7A5300", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/09-evaluation-and-testing/q11-red-teaming.svg")
