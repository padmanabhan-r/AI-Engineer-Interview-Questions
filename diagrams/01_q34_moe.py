"""LLM Fundamentals Q34: Mixture of Experts: a router sends each token to the top-2 of 8 expert FFNs."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 500, "Mixture of Experts: each token visits only a few experts", "Total parameters (capacity) grow while active parameters per token (compute) stay small.")
s.pill(82, 246, 100, 40, "Token x", "human")
s.arrow([(132, 246), (158, 246)], "human")
s.diamond(214, 246, 108, 84, "Router", "amber")
s.text(214, 306, "scores all 8, keeps", size=11, fill="#7A5300")
s.text(214, 321, "top 2, softmax gates", size=11, fill="#7A5300")

active = {2: "g_2", 5: "g_5"}
for i in range(1, 9):
    y = 108 + (i - 1) * 38
    on = i in active
    s.pill(400, y, 136, 30, f"Expert {i} · FFN", "compute" if on else "slate", size=12, weight=700 if on else 500)
    if on:
        s.arrow([(268, 246), (330, y)], "amber")
        s.text(284 if i == 2 else 300, 186 if i == 2 else 270, active[i], size=12, weight=700, fill="#7A5300", mono=True)
        s.arrow([(468, y), (516, 250 if i == 5 else 232)], "compute")
s.text(400, 400, "6 of 8 idle for this token", size=11.5, fill="#667085")
s.box(518, 206, 150, 76, "Weighted sum", ["Σ g_i · E_i(x)", "top-2 only"], "output", size=13.5, mono_detail=True)

# Mixtral facts
s.region(690, 86, 166, 192, "Mixtral 8x7B", "slate", dashed=False)
for k, line in enumerate(["8 experts per layer", "top-2 per token", "≈47B params total", "≈13B active per token", "attention is shared"]):
    s.text(706, 126 + k * 28, line, size=12, fill="#344054", anchor="start", weight=600 if k in (2, 3) else 400)
s.box(690, 294, 166, 76, "Not a memory win", ["all experts must", "stay loaded"], "fail", size=13)

s.box(36, 420, 404, 62, "Load balancing", ["aux loss, capacity limits or DeepSeek's", "bias adjustment stop routing collapse"], "amber", size=13)
s.box(452, 420, 404, 62, "Expert parallelism", ["all-to-all exchange in every MoE layer;", "gains shrink at small batch sizes"], "compute", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q34-moe.svg")
