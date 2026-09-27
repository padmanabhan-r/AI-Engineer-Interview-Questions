"""System design Q23: text-to-SQL over thousands of tables: retrieve from a semantic layer, validate before executing."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 400, "Text-to-SQL over thousands of tables", "Picking the right tables and metric definition is the hard part; nothing runs until it parses, dry-runs and fits the cost cap.")
s.cylinder(475, 112, 250, 56, "Semantic layer + query log", "data", size=12.5)
s.arrow([(475, 140), (475, 186)], "data")
# row 1
s.pill(88, 220, 120, 40, "Question", "human")
s.arrow([(148, 220), (174, 220)], "compute")
s.box(176, 188, 190, 64, "Disambiguate", ["metric, time range, grain"], "compute", size=13.5)
s.arrow([(366, 220), (390, 220)], "data")
s.box(392, 188, 166, 64, "Retrieve", ["5–15 tables, metrics,", "example queries"], "data", size=13.5)
s.arrow([(558, 220), (584, 220)], "model")
s.box(586, 188, 200, 64, "Generate", ["SQL or a metric query"], "model", size=13.5)
# row 2: validate, then execute as the user
s.arrow([(686, 252), (686, 300)], "model")
s.hexagon(686, 334, 230, 66, "Parse · allow-list\nEXPLAIN · cost cap", "amber", size=12.5)
s.arrow([(801, 334), (846, 334), (846, 220), (788, 220)], "fail")
s.text(838, 270, "error: repair,", size=11.5, fill="#8E2A23", weight=600, anchor="end")
s.text(838, 285, "1–2 tries", size=11.5, fill="#8E2A23", weight=600, anchor="end")
s.arrow([(571, 334), (512, 334)], "output", label="ok", label_dy=-11)
s.box(300, 300, 210, 68, "Execute read-only", ["as the user: row- and", "column-level security"], "compute", size=13.5)
s.arrow([(300, 334), (246, 334)], "output")
s.box(40, 302, 204, 64, "Result + SQL", ["+ tables and assumptions"], "output", size=13.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q23-text-to-sql.svg")
