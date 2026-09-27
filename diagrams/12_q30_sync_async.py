"""Infrastructure Q30: synchronous versus asynchronous inference, side by side."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 490, "Synchronous vs asynchronous inference", "Hold the connection when a user is waiting; hand back a job id for anything long, bulk or deadline-tolerant.")

def life(x, name, role, top=128, bottom=340, w=120):
    s.pill(x, top, w, 34, name, role, size=12.5)
    s.add(f'<line x1="{x}" y1="{top + 18}" x2="{x}" y2="{bottom}" stroke="{PALETTE[role][0]}" stroke-opacity="0.35" stroke-width="2" stroke-dasharray="4 5"/>')

def msg(y, x0, x1, label, role, dashed=False, mono=False):
    s.arrow([(x0, y), (x1 - (8 if x1 > x0 else -8), y)], role, dashed=dashed)
    s.text((x0 + x1) / 2, y - 12, label, size=11.5, fill=PALETTE[role][2], weight=600, mono=mono)

# sync
s.region(24, 84, 300, 382, "SYNC · a user is waiting", "compute", dashed=False)
life(84, "Client", "human", w=96)
life(254, "API + model", "model", w=120)
msg(184, 84, 254, "request", "human")
s.add(f'<rect x="248" y="198" width="12" height="96" rx="4" fill="{PALETTE["model"][0]}" fill-opacity="0.25"/>')
for i, y in enumerate([218, 244, 270, 296]):
    s.arrow([(248, y), (92, y)], "output", dashed=True, width=1.6)
s.text(169, 208, "token stream", size=11.5, fill="#275C1C", weight=600)
s.text(174, 322, "connection held open", size=11, fill="#667085", italic=True)
s.text(174, 372, "chat · copilots", size=11.5, fill="#1B418C", weight=600)
s.text(174, 390, "autocomplete · watched agent steps", size=11.5, fill="#1B418C", weight=600)
s.text(174, 420, "stream, so perceived", size=11, fill="#667085")
s.text(174, 436, "latency is TTFT", size=11, fill="#667085")

# async
s.region(340, 84, 536, 382, "ASYNC · a job, delivered later", "data", dashed=False)
life(420, "Client", "human", w=96)
life(620, "API + queue", "compute", w=124)
life(806, "GPU worker", "model", w=116)
msg(184, 420, 620, "POST /jobs", "human", mono=True)
msg(222, 620, 420, "202 Accepted, job_id", "compute", dashed=True)
msg(262, 806, 620, "pull job", "model")
s.add(f'<rect x="800" y="270" width="12" height="36" rx="4" fill="{PALETTE["model"][0]}" fill-opacity="0.25"/>')
msg(300, 800, 620, "store result", "model", dashed=True)
msg(334, 620, 420, "webhook, or client polls", "output", dashed=True)
s.text(608, 370, "documents · bulk extraction · evals · backfills · long agent tasks", size=11.5, fill="#0A5A51", weight=600)
chips = ["idempotency keys", "durable job state", "dead-letter queue", "backpressure"]
for i, c in enumerate(chips):
    s.pill(484 + (i % 2) * 248, 406 + (i // 2) * 36, 236, 28, c, "data", size=11.5, weight=600)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/12-ai-infrastructure-and-scalability/q30-sync-async.svg")
