"""Agents Q50: computer-use agents: screenshot in, one GUI action out, repeated on a sandboxed machine."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 548, "Computer-use agents: screenshot in, action out", "No API needed: the model reads pixels and emits coordinates; the harness acts on a sandboxed machine and captures the result.")

s.box(30, 140, 170, 100, "Multimodal model", ["chooses one action", "from the screenshot"], "model", size=13.5)
s.box(350, 140, 180, 100, "Harness", ["scales coordinates:", "model ↔ screen resolution"], "amber", size=13.5, detail=11)

# model <-> harness
s.arrow([(348, 164), (202, 164)], "amber")
s.text(275, 138, "task + recent\nscreenshots + history", size=11, fill="#7A5300", weight=600)
s.arrow([(202, 216), (348, 216)], "model")
s.text(275, 234, "click(x=412, y=233)", size=11, fill="#3B2596", mono=True)

# the sandboxed machine, with the click landing on it
s.region(642, 96, 228, 200, "Sandboxed VM", "slate", dashed=False)
s.add('<rect x="658" y="126" width="196" height="148" rx="8" fill="#FFFFFF" stroke="#98A2B3"/>')
s.add('<rect x="658" y="126" width="196" height="20" rx="8" fill="#E4E7EC"/>')
for i, c in enumerate(["#D0443A", "#C98A06", "#3E8E2F"]):
    s.add(f'<circle cx="{671 + i * 12}" cy="136" r="3.5" fill="{c}"/>')
for y, w in [(162, 130), (178, 100), (194, 150)]:
    s.add(f'<rect x="674" y="{y}" width="{w}" height="7" rx="3" fill="#E4E7EC"/>')
s.add('<rect x="730" y="220" width="84" height="24" rx="6" fill="#E7F0FF" stroke="#2F6FE4"/>')
s.add('<circle cx="772" cy="232" r="13" fill="none" stroke="#D6408F" stroke-width="2"/>'
      '<path d="M772,213 V251 M753,232 H791" stroke="#D6408F" stroke-width="1.6"/>')
s.text(712, 262, "click lands here", size=10.5, fill="#8A1F58")

s.arrow([(532, 216), (640, 216)], "amber")
s.text(586, 204, "act", size=11.5, fill="#7A5300", weight=600)
s.arrow([(640, 164), (532, 164)], "data")
s.text(586, 152, "new screenshot", size=11, fill="#0A5A51", weight=600)
s.text(440, 262, "repeat until done or a limit is hit", size=11.5, fill="#344054", weight=600)

# actions
s.text(30, 316, "Actions", size=12.5, weight=700, fill="#344054", anchor="start")
x = 100
for a in ["click at x, y", "type", "key press", "scroll"]:
    w = len(a) * 7.5 + 28
    s.add(f'<rect x="{x}" y="302" width="{w}" height="28" rx="14" fill="#F1ECFF" stroke="#6D4AE0"/>')
    s.text(x + w / 2, 316, a, size=11.5, fill="#3B2596", weight=600)
    x += w + 10
s.text(x + 6, 316, "via OS input automation or a browser driver", size=11, fill="#667085", anchor="start")

# what stays in context
s.text(30, 366, "Context each step", size=12.5, weight=700, fill="#344054", anchor="start")
x = 30
for i in range(3):
    s.add(f'<rect x="{x}" y="380" width="92" height="44" rx="6" fill="#F2F4F7" stroke="#98A2B3"/>')
    for j in range(3):
        s.add(f'<rect x="{x + 10}" y="{390 + j * 9}" width="{70 - j * 14}" height="4" rx="2" fill="#98A2B3"/>')
    x += 100
for i in range(3):
    s.add(f'<rect x="{x}" y="380" width="92" height="44" rx="6" fill="#E3F6F2" stroke="#0E8C7E"/>')
    s.add(f'<path d="M{x + 14},{414} L{x + 36},{394} L{x + 52},{408} L{x + 64},{398} L{x + 80},{414} Z" fill="#0E8C7E" opacity="0.55"/>')
    s.add(f'<circle cx="{x + 70}" cy="391" r="4" fill="#0E8C7E" opacity="0.7"/>')
    x += 100
s.text(168, 440, "older steps become text notes", size=11, fill="#344054")
s.text(468, 440, "only the last few screenshots", size=11, fill="#0A5A51")

s.box(30, 468, 840, 58, "Pitfall: on-screen text can inject instructions",
      ["isolate in a VM without the user's sessions · confirm purchases and sends · prefer APIs where they exist"], "fail", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q50-computer-use-agent.svg")
