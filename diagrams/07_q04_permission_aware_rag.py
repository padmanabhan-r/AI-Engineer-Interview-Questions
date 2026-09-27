"""System design Q4: enterprise RAG over 10M documents, with ACLs bound at index time and checked again live."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(900, 540, "Permission-aware enterprise RAG", "Permissions are a filter inside retrieval plus a live check on the final documents, never an instruction in the prompt.")

# ingest column
s.region(24, 80, 316, 440, "INGEST · change feeds", "data")
s.box(54, 110, 256, 64, "Source systems", ["SharePoint, Confluence, Drive, Jira"], "slate", size=13.5)
s.arrow([(182, 174), (182, 202)], "slate")
s.box(54, 204, 256, 64, "Connectors", ["content + ACL change feeds"], "data", size=13.5)
s.arrow([(182, 268), (182, 292)], "data")
s.cylinder(182, 330, 200, 74, "Principal store", "data")
s.text(182, 380, "user → groups", size=11.5, fill="#0A5A51", weight=600)
s.arrow([(54, 236), (40, 236), (40, 450), (80, 450)], "data")
s.cylinder(182, 450, 200, 74, "Sharded hybrid index", "data", size=12.5)
s.text(182, 500, "group IDs on every chunk", size=11.5, fill="#0A5A51", weight=600)

# query column
s.region(360, 80, 516, 440, "QUERY · per request", "compute", label_pos="tr")
s.pill(486, 150, 188, 42, "User via SSO", "human")
s.arrow([(486, 171), (486, 299)], "human")
s.box(392, 301, 188, 62, "Expand principals", ["user → group set"], "compute", size=13.5)
s.arrow([(282, 332), (390, 332)], "data", dashed=True)
s.arrow([(486, 363), (486, 419)], "compute")
s.box(392, 421, 188, 62, "Filtered search", ["ACL filter inside ANN"], "compute", size=13.5)
s.arrow([(282, 452), (390, 452)], "data", dashed=True)
s.arrow([(580, 452), (640, 452)], "compute", label="top-k", label_dy=-11)
s.hexagon(742, 452, 196, 66, "Live permission\ncheck", "fail", size=13)
s.text(742, 500, "closes sync lag · deny wins", size=11.5, fill="#8E2A23", weight=600)
s.arrow([(742, 419), (742, 363)], "fail")
s.box(648, 301, 188, 62, "LLM", ["answers with citations"], "model", size=14)
s.arrow([(742, 301), (742, 173)], "model")
s.pill(742, 150, 188, 42, "Cited answer", "output")
s.text(754, 238, "audit: query +", size=11.5, fill="#344054", anchor="start")
s.text(754, 256, "retrieved docs", size=11.5, fill="#344054", anchor="start")
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/07-ai-system-design/q04-permission-aware-rag.svg")
