"""LLMOps Q40: secrets never enter the model's context; the tool executor gets scoped tokens from a broker."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 470, "Keep secrets out of the model's context", "Anything in the prompt can be extracted by injection, so credentials live in the code that executes actions.")

# model side
s.region(28, 84, 250, 344, "MODEL CONTEXT · untrusted", "fail")
s.box(48, 160, 206, 88, "LLM", ["no credentials", "chooses the action only"], "model", size=15)
s.add('<rect x="48" y="290" width="206" height="112" rx="12" fill="#FFFFFF" stroke="#D0443A" stroke-dasharray="4 4"/>')
s.text(62, 310, "Never in here:", size=12, weight=700, fill="#8E2A23", anchor="start")
for i, l in enumerate(["API keys, connection strings", "secrets in the system prompt", "Authorization headers in", "tool results, logs or traces"]):
    s.text(62, 332 + i * 18, ("✗ " if i in (0, 1, 2) else "   ") + l, size=11.5, fill="#8E2A23", anchor="start")

# trust boundary
s.add('<line x1="304" y1="92" x2="304" y2="420" stroke="#344054" stroke-width="2" stroke-dasharray="7 5"/>')
s.text(304, 440, "trust boundary", size=11.5, weight=700, fill="#344054")

# executor side
s.region(330, 84, 546, 344, "EXECUTOR SIDE · credentials live here", "output")
s.box(404, 160, 176, 88, "Tool executor", ["runs the tool call"], "compute", size=14)
s.arrow([(254, 180), (402, 180)], "model")
s.arrow([(404, 228), (256, 228)], "compute")


def chip(x, y, label, color, w):
    s.add(f'<rect x="{x - w / 2}" y="{y - 10}" width="{w}" height="20" rx="10" fill="#FFFFFF" stroke="{color}" stroke-opacity="0.5"/>')
    s.text(x, y, label, size=11, weight=600, fill=color)


chip(329, 160, "tool call", "#3B2596", 70)
chip(329, 248, "result, secrets stripped", "#1B418C", 142)
s.hexagon(492, 344, 190, 64, "Credential broker", "amber")
s.arrow([(470, 248), (470, 310)], "amber")
s.text(462, 280, "check user\npermission", size=11, weight=600, fill="#7A5300", anchor="end")
s.box(680, 160, 172, 88, "External API", ["called with the", "user's scoped token"], "slate", size=14)
s.arrow([(587, 344), (766, 344), (766, 250)], "amber", label="scoped, short-lived token", label_at=0.35, label_dy=-11)
s.arrow([(678, 204), (582, 204)], "slate", label="response", label_dy=-11)
s.text(700, 402, "broker fetches at runtime via workload\nidentity from a secrets manager", size=11, fill="#275C1C", weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/08-llmops-and-production-ai/q40-secrets-credential-broker.svg")
