"""LLMOps Q26: the LLM gateway pattern: one internal service between every app and every provider."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 470, "The LLM gateway pattern", "One internal API in front of every provider: keys, routing, quotas, cost and logs live in one place.")

# callers
apps = [(130, "App A"), (220, "App B"), (310, "Agent service")]
for y, lab in apps:
    s.pill(106, y, 150, 40, lab, "human", size=12.5)
    s.arrow([(181, y), (254, 220 + (y - 220) * 0.35)], "human")
s.text(106, 364, "virtual keys:", size=11.5, weight=700, fill="#1F3864")
s.text(106, 382, "team, budget,\nallowed models", size=11, fill="#1F3864")

# the gateway
st, f, ink = PALETTE["compute"]
s.add(f'<rect x="256" y="86" width="388" height="330" rx="16" fill="{f}" stroke="{st}" stroke-width="2"/>')
s.add(f'<rect x="256" y="86" width="8" height="330" rx="4" fill="{st}"/>')
s.text(450, 112, "LLM gateway", size=16, weight=700, fill=ink)
s.text(450, 132, "stateless, multi-zone", size=11.5, fill=ink, opacity=0.8)
# logical name resolution
s.add('<rect x="280" y="150" width="344" height="62" rx="10" fill="#FFFFFF" stroke="#98A2B3"/>')
s.text(296, 170, "chat-default", size=12.5, weight=700, fill="#3B2596", anchor="start", mono=True)
s.text(400, 170, "→ pinned provider model", size=12, fill="#344054", anchor="start")
s.text(296, 194, "fallback order + retries", size=11.5, fill="#344054", anchor="start")
# capability chips
chips = ["auth, virtual keys", "routing", "fallbacks", "quotas", "cost per team", "caching",
         "logs + trace ID", "guardrails", "format translation", "streaming"]
for i, lab in enumerate(chips):
    r, c = divmod(i, 2)
    x, y = 280 + c * 176, 228 + r * 36
    role = ["compute", "amber", "data", "model", "slate"][r]
    s.pill(x + 82, y + 13, 164, 28, lab, role, size=11.5)

# providers
provs = [(150, "Provider 1", "model"), (250, "Provider 2", "model"), (350, "Self-hosted engine", "data")]
for y, lab, role in provs:
    s.box(716, y - 30, 158, 60, lab, [], role, size=13)
    s.arrow([(646, 250 + (y - 250) * 0.35), (714, y)], role)
s.text(795, 412, "provider keys never\nleave the gateway", size=11.5, weight=600, fill="#3B2596")

s.text(450, 446, "Trade-off: an extra hop (milliseconds) and a critical dependency. Keep business logic out of it.", size=11.5, fill="#667085")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/08-llmops-and-production-ai/q26-llm-gateway.svg")
