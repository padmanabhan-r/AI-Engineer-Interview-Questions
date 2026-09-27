"""System design Q39: a multi-tenant chatbot platform: shared runtime, per-tenant config, tenant ID from auth on every call."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 560, "A multi-tenant chatbot platform", "One shared runtime, per-tenant config; the tenant ID comes from auth and scopes every call, never from the model.")

# row 1: entry
s.box(36, 86, 172, 56, "Channels", ["widget, WhatsApp, Slack"], "slate", size=13.5)
s.arrow([(208, 114), (232, 114)], "human")
s.hexagon(334, 114, 200, 56, "Tenant ID from auth", "human", size=13)
s.text(334, 154, "API key or domain", size=11, fill="#1F3864", weight=600)
s.arrow([(434, 114), (462, 114)], "amber")
s.hexagon(540, 114, 150, 56, "Per-tenant\nquotas", "amber", size=13)
s.arrow([(540, 142), (540, 188)], "compute")

# row 2: the shared orchestrator, fed by tenant config and platform guardrails
s.box(360, 190, 220, 62, "Shared orchestrator", ["one runtime, all tenants"], "compute", size=14)
s.cylinder(186, 221, 190, 62, "Tenant config (versioned)", "data", size=12.5)
s.text(186, 266, "persona, instructions, knowledge, tools", size=11, fill="#0A5A51")
s.arrow([(281, 221), (358, 221)], "data")
s.hexagon(726, 221, 210, 56, "Platform guardrails", "fail", size=13)
s.text(726, 261, "tenants cannot override", size=11, fill="#8E2A23", weight=600)
s.arrow([(621, 221), (582, 221)], "fail")

# the tenant-scoped bus
navy = PALETTE["human"][0]
s.add(f'<path d="M470,252 V292 M140,292 H768" stroke="{navy}" stroke-width="3" fill="none" stroke-linecap="round"/>')
for x in (140, 352, 562, 768):
    s.arrow([(x, 292), (x, 314)], "human", width=2)
s.box(40, 316, 200, 60, "Retrieval", ["mandatory tenant filter"], "data", size=13.5)
s.box(254, 316, 196, 60, "Tenant tools", ["vault credentials"], "amber", size=13.5)
s.box(464, 316, 196, 60, "Model gateway", ["fair queuing"], "model", size=13.5)
s.box(674, 316, 188, 60, "Metering", ["per-tenant billing"], "slate", size=13.5)
s.text(484, 280, "every call carries tenant_id", size=11.5, fill=navy, weight=700, anchor="start")

# isolation tiers under retrieval
s.arrow([(140, 376), (140, 410)], "data")
s.region(24, 398, 500, 138, "", "data", dashed=False)
s.text(44, 426, "Small tenants: a namespace each", size=12, weight=700, fill="#0A5A51", anchor="start")
for i, t in enumerate(["A", "B", "C"]):
    s.cylinder(74 + i * 66, 482, 54, 56, t, "data", size=12.5)
s.text(262, 488, "…", size=16, fill="#0A5A51", weight=700)
s.text(300, 426, "Large or regulated", size=12, weight=700, fill="#3B2596", anchor="start")
s.cylinder(400, 482, 188, 56, "Dedicated index", "model", size=12.5)
s.text(400, 522, "own capacity or region, priced", size=11, fill="#3B2596")

# leak paths
s.region(544, 398, 332, 138, "Close the leak paths", "fail", dashed=False)
rows = [("•", "No tenant ID taken from prompts or tool args"), ("•", "tenant_id in every cache key and log"), ("•", "Tenant test set on every config change")]
for i, (b, line) in enumerate(rows):
    s.text(564, 446 + i * 30, f"{b}  {line}", size=12, fill="#8E2A23", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q39-multi-tenant-chatbot.svg")
