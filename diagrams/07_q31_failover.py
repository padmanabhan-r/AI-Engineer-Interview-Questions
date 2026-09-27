"""System design Q31: failover and fallback: a bounded chain of fallbacks, a circuit breaker, and error classification."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 500, "Failover and fallback for AI systems", "Treat the provider as unreliable: retry only what is retryable, then step down a fallback chain evaluated in advance.")
# the fallback staircase
s.pill(86, 112, 116, 40, "Request", "human")
s.arrow([(144, 112), (176, 112)], "compute")
s.box(178, 86, 190, 52, "Primary model", ["region 1"], "compute", size=13.5)
s.arrow([(230, 138), (230, 178)], "fail")
s.text(240, 158, "timeout · 429 · 5xx", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.pill(330, 198, 250, 40, "Backoff + jitter, bounded", "amber", size=12.5)
s.arrow([(290, 218), (290, 252)], "fail")
s.text(300, 236, "still failing", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.box(238, 254, 190, 52, "Same model", ["region 2"], "compute", size=13.5)
s.arrow([(350, 306), (350, 340)], "fail")
s.text(360, 324, "failing", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.box(298, 342, 210, 52, "Other provider", ["adapted prompt variant"], "pink", size=13.5)
s.arrow([(410, 394), (410, 428)], "fail")
s.text(420, 412, "failing", size=11.5, fill="#8E2A23", weight=600, anchor="start")
s.pill(452, 450, 190, 40, "Degraded mode", "slate")
# circuit breaker skips the dead link
s.arrow([(86, 132), (86, 246)], "fail", dashed=True)
s.hexagon(98, 280, 150, 60, "Circuit\nbreaker", "fail", size=12.5)
s.arrow([(173, 280), (236, 280)], "fail", dashed=True)
s.text(98, 324, "per provider-model pair:", size=11, fill="#8E2A23", anchor="middle")
s.text(98, 338, "open → skip to fallback,", size=11, fill="#8E2A23", anchor="middle")
s.text(98, 352, "probe with a trickle", size=11, fill="#8E2A23", anchor="middle")

# classify errors panel
s.region(574, 84, 302, 392, "Classify before retrying", "slate", dashed=False)
def chips(y0, head, role, rows):
    st, f, ink = PALETTE[role]
    s.text(592, y0, head, size=11.5, weight=700, fill=ink, anchor="start")
    for i, r in enumerate(rows):
        y = y0 + 26 + i * 34
        s.add(f'<rect x="590" y="{y - 14}" width="270" height="28" rx="8" fill="{f}" stroke="{st}" stroke-opacity="0.6"/>')
        s.text(602, y, r, size=11.5, fill=ink, anchor="start")
chips(126, "RETRY", "output", ["429, honouring Retry-After", "5xx", "timeout: first token, inter-token gap"])
chips(252, "NEVER RETRY", "fail", ["400s", "context overflow", "policy refusal"])
chips(378, "GUARDS", "amber", ["retry budget, say 10% of traffic", "idempotency keys on tool calls"])
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q31-failover.svg")
