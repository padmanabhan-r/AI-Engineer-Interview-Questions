"""Agents Q7: function calling as a message sequence; the model requests, your code executes."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 600, "Function calling: the model asks, your code acts", "The model emits a tool name plus JSON arguments and stops; your code runs the tool and sends the result back.")

L = {"code": 140, "model": 450, "tool": 760}
heads = [("code", "Your code", "compute"), ("model", "Model API", "model"), ("tool", "Tool", "amber")]
for k, label, role in heads:
    s.box(L[k] - 80, 84, 160, 40, label, (), role, size=13.5)
    s.add(f'<line x1="{L[k]}" y1="124" x2="{L[k]}" y2="486" stroke="{PALETTE[role][0]}" stroke-width="1.5" stroke-dasharray="4 5" opacity="0.6"/>')


def msg(y, a, b, label, role, dashed=False, mono=True, note="", lx=None):
    x0, x1 = L[a], L[b]
    cx = lx or (x0 + x1) / 2
    d = 1 if x1 > x0 else -1
    s.arrow([(x0 + d * 6, y), (x1 - d * 8, y)], role, dashed=dashed)
    s.text(cx, y - 12, label, size=11.5, fill=PALETTE[role][2], mono=mono)
    if note:
        s.text(cx, y + 13, note, size=10.5, fill="#667085")


msg(160, "code", "model", "messages + tool schemas", "compute", mono=False, note="names, descriptions, JSON Schemas")
msg(214, "model", "code", 'tool_call get_weather(city="Paris")', "model", dashed=True, note="+ a stop reason; nothing has run yet")
s.hexagon(L["code"], 262, 120, 34, "validate args", "amber", size=12)
msg(314, "code", "tool", 'get_weather("Paris")', "compute", note="your code executes", lx=605)
msg(370, "tool", "code", "18 C, cloudy", "amber", dashed=True, lx=605)
msg(420, "code", "model", "messages + tool result", "compute", mono=False, note="linked by call ID")
msg(470, "model", "code", '"It is 18 C and cloudy in Paris"', "output", dashed=True)

s.box(40, 510, 390, 64, "Your code's job, not the model's", ["validation · permissions · timeouts · idempotency"], "compute", size=13)
s.box(450, 510, 390, 64, "Strict mode", ["constrained decoding: arguments always parse,", "values can still be wrong"], "slate", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q07-function-calling.svg")
