"""Must Know Q3: MCP: a host runs one client per server; servers expose tools, resources and prompts over JSON-RPC."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 524, "MCP: wrap a tool once, any host can use it", "The model still does ordinary function calling; MCP standardizes how tools are discovered and invoked.")
# host
s.region(24, 84, 330, 250, "HOST · the AI app", "human")
s.box(44, 176, 146, 70, "LLM", ["ordinary function", "calling"], "model")
s.arrow([(190, 198), (224, 160)], "compute")
s.arrow([(190, 224), (224, 262)], "compute")
s.pill(282, 150, 116, 38, "MCP client", "compute", size=12.5)
s.pill(282, 272, 116, 38, "MCP client", "compute", size=12.5)
s.text(282, 312, "one client per server", size=11, fill="#1F3864", weight=600)

# servers
for y, where, wire in [(100, "MCP server · local", "JSON-RPC 2.0 · stdio"), (222, "MCP server · remote", "JSON-RPC 2.0 · HTTP + OAuth")]:
    s.box(560, y, 296, 100, where, ["", ""], "data", size=13.5)
    for cx, w, lab in [(610, 76, "tools"), (706, 100, "resources"), (804, 84, "prompts")]:
        s.pill(cx, y + 66, w, 26, lab, "data", size=11)
    cy = 150 if y == 100 else 272
    mid = s._marker(PALETTE["compute"][0])
    s.add(f'<path d="M343,{cy} L556,{cy}" fill="none" stroke="{PALETTE["compute"][0]}" stroke-width="2" marker-start="url(#{mid})" marker-end="url(#{mid})"/>')
    s.text(456, cy - 13, wire, size=11.5, weight=600, fill="#1B418C")
s.text(706, 336, "tools take JSON Schema args; resources are readable data", size=11, fill="#667085")

# the exchange
s.region(24, 354, 832, 150, "The exchange", "slate", dashed=False)
rows = [("compute", "host → server", "initialize, tools/list"),
        ("data", "server → host", "tool names + JSON Schemas"),
        ("compute", "host → server", "tools/call with arguments"),
        ("data", "server → host", "result, into the model's context")]
for i, (role, who, msg) in enumerate(rows):
    y = 394 + i * 28
    col, fill, ink = PALETTE[role]
    s.add(f'<rect x="44" y="{y - 12}" width="480" height="24" rx="7" fill="{fill}" stroke="{col}" stroke-opacity="0.5"/>')
    s.text(56, y + 1, who, size=11.5, weight=700, fill=ink, anchor="start")
    s.text(170, y + 1, msg, size=11.5, fill=ink, anchor="start", mono=True)
s.arrow([(526, 422), (560, 422)], "fail")
s.arrow([(526, 478), (560, 478)], "fail")
s.box(562, 404, 274, 88, "Injection risk", ["descriptions and results enter the", "context: an untrusted server can", "inject instructions"], "fail", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/00-must-know/q03-mcp.svg")
