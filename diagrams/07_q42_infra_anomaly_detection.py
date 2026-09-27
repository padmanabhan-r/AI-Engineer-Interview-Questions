"""System design Q42: infrastructure anomaly detection: cheap per-series detectors, topology correlation, and an LLM that only explains."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 472, "Anomaly detection for cloud infrastructure", "Statistics detect, the service graph correlates, and the LLM only explains the incident it is handed.")

# detect
s.region(24, 78, 578, 214, "DETECT · streaming, 500k points/s", "compute")
s.box(40, 150, 130, 76, "Telemetry", ["metrics, logs,", "traces"], "slate", size=13.5)
s.arrow([(170, 188), (192, 188)], "compute")
s.pill(240, 188, 92, 48, "Stream\ningest", "compute", size=12.5)
s.arrow([(286, 188), (300, 188), (300, 140), (318, 140)], "compute")
s.arrow([(300, 188), (300, 238), (318, 238)], "compute")
s.box(320, 106, 264, 68, "Per-series detectors", ["seasonal baseline, then robust", "z-score (median, MAD)"], "compute", size=13.5)
s.box(320, 206, 264, 64, "Log templates", ["cluster lines, rate anomalies"], "compute", size=13.5)

# correlate
s.region(618, 78, 258, 214, "CORRELATE", "amber")
s.arrow([(584, 140), (636, 140)], "amber")
s.arrow([(584, 238), (608, 238), (608, 168), (636, 168)], "amber")
s.box(638, 116, 222, 72, "Correlate", ["topology, time, deploys"], "amber", size=13.5)
s.cylinder(706, 252, 150, 62, "Service graph\n+ changes", "data", size=12)
s.arrow([(706, 221), (706, 190)], "data")

# explain
s.region(24, 310, 852, 140, "EXPLAIN · only after correlation", "model", label_pos="br")
s.arrow([(826, 188), (826, 346)], "amber")
s.pill(790, 368, 132, 44, "One incident", "amber", size=12.5)
s.arrow([(724, 368), (646, 368)], "model")
s.box(410, 330, 234, 76, "LLM explainer", ["summary + ranked RCA hypotheses,", "citing graphs, templates, deploy ID"], "model", size=13.5)
s.arrow([(410, 368), (320, 368)], "human")
s.pill(250, 368, 136, 44, "On-call SRE", "human", size=12.5)
s.text(527, 424, "never acts on production", size=11.5, fill="#3B2596", weight=600)
s.arrow([(216, 346), (216, 300), (452, 300), (452, 272)], "human", dashed=True)
s.text(226, 324, "feedback labels", size=11.5, fill="#1F3864", weight=600, anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q42-infra-anomaly-detection.svg")
