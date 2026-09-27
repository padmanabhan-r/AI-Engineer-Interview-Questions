"""System design Q8: an on-device assistant, a local 4-bit model with LoRA adapters and consented escalation to the cloud."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 520, "An on-device AI assistant", "A small 4-bit model runs locally on private data; only hard requests go to the cloud, and only with consent.")

s.region(24, 80, 600, 420, "ON DEVICE · 8–16 GB shared RAM", "data")
s.region(644, 80, 232, 420, "CLOUD", "slate", label_pos="tr")

s.pill(118, 150, 150, 42, "App surface", "human")
s.arrow([(193, 150), (238, 150)], "human")
s.diamond(318, 150, 156, 84, "Local or\ncloud?", "amber", size=12.5)

# local path
s.arrow([(318, 192), (318, 246)], "data")
s.text(328, 220, "local", size=11.5, fill="#0A5A51", weight=600, anchor="start")
s.box(208, 248, 220, 76, "Local runtime", ["4-bit 1–4B model on the NPU", "memory-mapped weights"], "model", size=14)
s.arrow([(428, 286), (458, 286)], "model")
s.box(460, 254, 146, 64, "Task LoRA", ["adapters"], "model", size=13.5)
s.cylinder(110, 286, 140, 80, "Local index", "data", size=12.5)
s.text(110, 344, "messages, notes", size=11.5, fill="#0A5A51", weight=600)
s.arrow([(180, 286), (206, 286)], "data")

# cloud path
s.arrow([(396, 150), (668, 150)], "slate", label="hard task, with consent", label_dy=-11)
s.box(670, 118, 190, 64, "Cloud model", ["stateless"], "slate", size=14)
s.box(670, 336, 190, 64, "Signed updates", ["over the air"], "slate", size=13.5)
s.arrow([(670, 368), (318, 368), (318, 326)], "slate", dashed=True)

# budget panel
s.add('<rect x="44" y="400" width="562" height="84" rx="10" fill="#FFFFFF" stroke="#D0D5DD"/>')
s.text(60, 420, "Budget", size=12.5, weight=700, fill="#1F3864", anchor="start")
s.text(60, 442, "3B at 4 bits ≈ 1.5 GB + KV + runtime;  phone budget ≈ 1.5–3 GB", size=12, fill="#344054", anchor="start")
s.text(60, 464, "tok/s ≲ bandwidth ÷ model bytes:  60 GB/s ÷ 1.6 GB ≈ 35 tok/s", size=12, fill="#344054", anchor="start", mono=True)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q08-on-device-assistant.svg")
