"""Infrastructure Q22: autoscaling LLM serving as a control loop on engine load signals."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(880, 430, "Autoscaling LLM serving", "Scale on load signals with targets from the latency SLO; a GPU replica takes minutes to start, so keep warm capacity.")
# the control loop
s.region(24, 84, 492, 322, "CONTROL LOOP", "slate", dashed=False)
s.box(48, 124, 190, 70, "Inference replicas", ["vLLM exports load"], "model", size=13.5)
s.box(304, 124, 190, 70, "Prometheus", ["queue, KV usage, TTFT"], "data", size=13.5)
s.box(304, 310, 190, 70, "KEDA or HPA", ["target ~70–80% of the", "SLO-safe concurrency"], "compute", size=13.5)
s.box(48, 310, 190, 70, "Cluster autoscaler", ["adds GPU nodes"], "amber", size=13.5)
s.arrow([(238, 159), (300, 159)], "model", label="metrics", label_dy=-11)
s.arrow([(399, 194), (399, 306)], "data")
s.text(409, 250, "scale signal", size=11.5, fill="#0A5A51", weight=600, anchor="start")
s.arrow([(304, 345), (242, 345)], "compute", label="replicas", label_dy=-11)
s.arrow([(143, 310), (143, 198)], "amber")
s.text(133, 250, "GPU nodes", size=11.5, fill="#7A5300", weight=600, anchor="end")
s.text(270, 252, "up fast,", size=12, fill="#344054", weight=700)
s.text(270, 270, "down slowly", size=12, fill="#344054", weight=700)

# which signals
def chip(y, role, head, lines):
    c, f, ink = PALETTE[role]
    h = 39 + 17 * len(lines)
    s.add(f'<rect x="540" y="{y}" width="316" height="{h}" rx="12" fill="{f}" stroke="{c}" stroke-width="1.6"/>')
    s.text(558, y + 20, head, size=13, weight=700, fill=ink, anchor="start")
    for i, l in enumerate(lines):
        s.text(558, y + 42 + i * 17, l, size=11.5, fill=ink, anchor="start")
    return y + h + 14

y = chip(84, "output", "Scale on", ["queue depth", "in-flight requests per replica", "KV-cache usage"])
y = chip(y, "fail", "Never on", ["CPU or raw GPU utilization:", "nvidia-smi reads ~100% under light decode"])
y = chip(y, "amber", "Because replicas start in minutes", ["minimum replicas, warm node pool, cached", "weights; pre-scale for known peaks; drain", "in-flight streams before removing one"])
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/12-ai-infrastructure-and-scalability/q22-autoscaling.svg")
