"""LLM Fundamentals Q73: Recursive Language Models: the input lives in a REPL; the root model writes code and calls itself on slices."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 490, "Recursive Language Models: the input lives in a REPL", "An inference strategy, not an architecture: the model writes code to split the input and calls itself on the pieces.")
s.pill(150, 104, 150, 36, "Query", "human")
s.arrow([(150, 122), (150, 138)], "human")
s.box(50, 140, 200, 88, "Root LM", ["sees the query and what its", "code prints, never the", "whole input"], "model", size=14)

s.region(380, 94, 476, 156, "PYTHON REPL · sandbox", "compute", label_pos="tr")
s.cylinder(490, 178, 150, 76, "context", "data")
s.text(490, 232, "the large input, as a variable", size=11, fill="#0A5A51")
s.box(598, 130, 240, 84, "The root's code", ["grep  → a lookup", "iterate  → an aggregation", "chosen per query"], "slate", size=13, mono_detail=True)
s.arrow([(250, 160), (378, 160)], "model")
s.text(314, 148, "code: peek, split", size=11.5, weight=600, fill="#3B2596")
s.arrow([(410, 206), (252, 206)], "compute")
s.text(312, 222, "small printed result", size=11.5, weight=600, fill="#1B418C")

# sub-calls as a stack of cards
for k in (2, 1):
    s.add(f'<rect x="{70 + 8 * k}" y="{298 - 8 * k}" width="200" height="72" rx="12" fill="#F1ECFF" stroke="#6D4AE0" stroke-opacity="0.5" stroke-width="1.4"/>')
s.box(70, 298, 200, 72, "Sub-LM calls", ["one small, focused slice each;", "can themselves recurse"], "model", size=13.5)
s.arrow([(120, 228), (120, 280)], "model")
s.text(112, 248, "query on", size=11, weight=600, fill="#3B2596", anchor="end")
s.text(112, 262, "each slice", size=11, weight=600, fill="#3B2596", anchor="end")
s.arrow([(214, 280), (214, 230)], "model", dashed=True)
s.text(224, 256, "short answers", size=11, weight=600, fill="#3B2596", anchor="start")

# one run
s.region(380, 270, 476, 186, "One run", "slate", dashed=False)
rows = [("model", "Root → REPL", "code to peek at and split the input"),
        ("compute", "REPL → Root", "a small printed result"),
        ("model", "Root → Sub-LM", "the query on each slice"),
        ("model", "Sub-LM → Root", "short answers"),
        ("output", "Root → REPL", "combine and verify")]
for i, (role, who, msg) in enumerate(rows):
    y = 312 + i * 29
    col, fill, ink = PALETTE[role]
    s.add(f'<rect x="398" y="{y - 12}" width="440" height="24" rx="7" fill="{fill}" stroke="{col}" stroke-opacity="0.5"/>')
    s.text(410, y + 1, who, size=11.5, weight=700, fill=ink, anchor="start")
    s.text(530, y + 1, msg, size=11.5, fill=ink, anchor="start")

s.box(40, 390, 320, 80, "Trade-offs", ["cost and latency vary with the plan;", "needs a sandbox; weak models split", "poorly; plain lookups: retrieval is cheaper"], "fail", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q73-rlm.svg")
