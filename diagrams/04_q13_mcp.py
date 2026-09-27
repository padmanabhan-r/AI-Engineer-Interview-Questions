"""Agents Q13: MCP: a host runs one client per server over JSON-RPC; M x N integrations become M + N."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 560, "MCP: one protocol between AI apps and tools", "Wrap a tool once as an MCP server and any MCP client can use it; the host hands its tools to the model as ordinary functions.")


def biarrow(pts, role):
    c = PALETTE[role][0]
    m = s._marker(c)
    d = "M" + " L".join(f"{x},{y}" for x, y in pts)
    s.add(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" marker-start="url(#{m})" marker-end="url(#{m})"/>')


# host app
s.region(30, 84, 396, 236, "Host app · IDE, chat app, agent", "compute")
s.box(52, 160, 160, 80, "LLM", ["gets MCP tools as", "function definitions"], "model")
s.add(f'<path d="M212,200 H232 M232,148 V252" fill="none" stroke="{PALETTE["model"][0]}" stroke-width="2"/>')
s.arrow([(232, 148), (252, 148)], "model")
s.arrow([(232, 252), (252, 252)], "model")
s.box(254, 118, 150, 60, "MCP client 1", ["one per server"], "compute", size=13.5)
s.box(254, 222, 150, 60, "MCP client 2", ["one per server"], "compute", size=13.5)

# transports
biarrow([(410, 148), (556, 148)], "data")
s.text(483, 134, "stdio", size=11.5, fill="#0A5A51", weight=600, mono=True)
s.text(483, 163, "local", size=10.5, fill="#667085")
biarrow([(410, 252), (556, 252)], "pink")
s.text(483, 238, "Streamable HTTP", size=11.5, fill="#8A1F58", weight=600)
s.text(483, 267, "remote, OAuth", size=10.5, fill="#667085")
s.text(483, 200, "JSON-RPC 2.0", size=11.5, fill="#344054", weight=700, mono=True)

# servers
s.box(562, 110, 284, 76, "Local server", ["files, git"], "data")
s.box(562, 214, 284, 76, "Remote server", ["a SaaS API"], "pink")
s.text(704, 310, "each exposes: tools · resources · prompts", size=11.5, fill="#344054", weight=600)

# M x N -> M + N
s.region(30, 344, 396, 190, "Without MCP · M × N integrations", "fail", dashed=False)
s.region(450, 344, 396, 190, "With MCP · M + N", "output", dashed=False)
ys = [398, 443, 488]
for y in ys:
    for y2 in ys:
        s.add(f'<line x1="118" y1="{y}" x2="332" y2="{y2}" stroke="#D0443A" stroke-opacity="0.55" stroke-width="1.6"/>')
    s.add(f'<line x1="538" y1="{y}" x2="632" y2="443" stroke="#3E8E2F" stroke-opacity="0.7" stroke-width="1.6"/>')
    s.add(f'<line x1="672" y1="443" x2="744" y2="{y}" stroke="#3E8E2F" stroke-opacity="0.7" stroke-width="1.6"/>')
for y in ys:
    s.pill(86, y, 64, 28, "App", "slate", size=12)
    s.pill(364, y, 64, 28, "Tool", "data", size=12)
    s.pill(506, y, 64, 28, "App", "slate", size=12)
    s.pill(786, y, 84, 28, "Server", "data", size=12)
s.add('<rect x="628" y="383" width="48" height="120" rx="14" fill="#E8F5E1" stroke="#3E8E2F" stroke-width="1.6"/>')
s.text(652, 443, "MCP", size=12, weight=700, fill="#275C1C")
s.text(228, 516, "every app re-integrates every tool", size=11, fill="#8E2A23")
s.text(648, 516, "each app speaks MCP once; each tool is wrapped once", size=11, fill="#275C1C")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/04-ai-agents-and-agentic-systems/q13-mcp.svg")
