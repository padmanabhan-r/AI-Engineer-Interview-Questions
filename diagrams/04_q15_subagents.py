"""Agents Q15: subagents: fresh context windows that return only a condensed result to the parent."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 490, "Subagents: delegate the reading, keep the summary", "Each subagent burns its own fresh context window; the parent's context grows only by the short result it returns.")


def bar(x, y, w, segs, label):
    """A context window: segments of (fraction, role, text)."""
    s.add(f'<rect x="{x}" y="{y}" width="{w}" height="24" rx="6" fill="#FFFFFF" stroke="#D0D5DD"/>')
    cx = x
    for frac, role, t in segs:
        sw = w * frac
        col, fill, ink = PALETTE[role]
        s.add(f'<rect x="{cx + 1:.1f}" y="{y + 1}" width="{sw - 2:.1f}" height="22" rx="5" fill="{col}" fill-opacity="0.75"/>')
        if t:
            s.text(cx + sw / 2, y + 12, t, size=10.5, fill="#FFFFFF", weight=700)
        cx += sw
    s.text(x, y + 38, label, size=11, fill="#667085", anchor="start")


# parent
s.box(40, 104, 220, 256, "", (), "model")
s.text(153, 146, "Parent agent", size=14, weight=700, fill="#3B2596")
s.text(153, 170, "delegates with a tool call", size=11.5, fill="#3B2596", opacity=0.85)
s.text(153, 192, "task(prompt, agent_type)", size=11, fill="#3B2596", mono=True)
s.text(62, 262, "its context window", size=11.5, fill="#3B2596", weight=700, anchor="start")
bar(62, 274, 180, [(0.5, "model", "own work"), (0.1, "data", ""), (0.1, "compute", "")], "+ two short results only")

# subagent A
s.arrow([(260, 134), (448, 134)], "model")
s.text(354, 120, "find callers of auth API", size=11.5, fill="#3B2596", weight=600)
s.box(450, 104, 200, 84, "Subagent A", ["read-only tools"], "data")
s.arrow([(448, 168), (262, 168)], "data", dashed=True)
s.text(354, 184, "short summary", size=11.5, fill="#0A5A51", weight=600)
bar(670, 126, 180, [(0.95, "data", "reads fifty files")], "its own fresh context window")

# subagent B
s.arrow([(260, 290), (448, 290)], "model")
s.text(354, 276, "summarize failing tests", size=11.5, fill="#3B2596", weight=600)
s.box(450, 260, 200, 84, "Subagent B", ["shell"], "compute")
s.arrow([(448, 324), (262, 324)], "compute", dashed=True)
s.text(354, 340, "3 failures + causes", size=11.5, fill="#1B418C", weight=600)
bar(670, 282, 180, [(0.9, "compute", "test output")], "its own fresh context window")

s.box(40, 400, 810, 60, "The child knows only its brief", ["state the goal, constraints and answer shape; keep writes with one agent"], "amber", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q15-subagents.svg")
