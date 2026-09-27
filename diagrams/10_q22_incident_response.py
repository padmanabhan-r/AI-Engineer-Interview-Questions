"""Safety Q22: the AI incident response lifecycle, with the containment levers and deadlines from the answer."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg, PALETTE

s = Svg(900, 440, "AI incident response lifecycle", "Harmful or wrong outputs are incidents too; build the containment levers, and drill them, before one happens.")
W, H = 176, 74
X = [36, 250, 464, 678]
Y1, Y2 = 96, 236
row1 = [("Detected", ["guardrail block spikes,", "canaries, drift, reports"], "amber"),
        ("Triaged", ["severity assigned", "SEV1: harm, PII leak"], "human"),
        ("Contained", ["kill switch or rollback", "flag · fallback path"], "fail"),
        ("Investigated", ["scope the affected", "decisions"], "compute")]
for x, (t, l, r) in zip(X, row1):
    s.box(x, Y1, W, H, t, l, r, size=14, detail=11)
for i in range(3):
    s.arrow([(X[i] + W, Y1 + H / 2), (X[i + 1] - 2, Y1 + H / 2)], row1[i + 1][2])
s.arrow([(X[3] + W / 2, Y1 + H), (X[3] + W / 2, Y2 - 2)], "model")
row2 = [(X[3], "Remediated", ["fix + regression eval"], "model"),
        (X[2], "Notified", ["users, regulators", "GDPR 72 h · AI Act ~15 days"], "pink"),
        (X[1], "Reviewed", ["blameless post-mortem"], "output")]
for x, t, l, r in row2:
    s.box(x, Y2, W, H, t, l, r, size=14, detail=11)
s.arrow([(X[3], Y2 + H / 2), (X[2] + W + 2, Y2 + H / 2)], "pink")
s.arrow([(X[2], Y2 + H / 2), (X[1] + W + 2, Y2 + H / 2)], "output")
# every incident becomes an eval case, which feeds detection
s.arrow([(X[1], Y2 + H / 2), (X[0] + W / 2, Y2 + H / 2), (X[0] + W / 2, Y1 + H + 2)], "output", dashed=True)
s.text((X[0] + W / 2 + X[1]) / 2 + 10, Y2 + H / 2 - 11, "new eval case", size=11.5, weight=600, fill=PALETTE["output"][2])

s.region(24, 344, 852, 76, "Before any incident", "slate", dashed=False)
s.text(44, 380, "Build a feature flag per AI feature, one-click rollback of model, prompt and guardrail versions, and per-tool kill switches.",
       size=11.5, fill="#344054", anchor="start")
s.text(44, 400, "A kill switch nobody has flipped in a drill often fails in a real incident: run tabletop exercises.",
       size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/10-ai-safety-ethics-and-responsible-ai/q22-incident-response.svg")
