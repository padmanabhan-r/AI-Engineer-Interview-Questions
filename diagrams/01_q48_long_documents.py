"""LLM Fundamentals Q48: long documents: decide part vs all, then retrieve or map-reduce; bigger context is the fallback."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(880, 490, "Long documents: retrieve a part, or map-reduce the whole", "Decide whether the task needs part of the document or all of it; a bigger window is the fallback, not the default.")
s.pill(96, 118, 140, 40, "Long document", "data", size=12.5)
s.arrow([(166, 118), (198, 118)], "data")
s.diamond(270, 118, 140, 84, "Part or all?", "amber")
s.arrow([(340, 118), (398, 118)], "compute", label="part", label_dy=-11)
s.box(400, 86, 210, 64, "Retrieve relevant chunks", ["lookup, question answering"], "compute", size=13)
s.box(636, 86, 220, 64, "Bigger context window", ["the fallback: pricier per call,", "weaker in the middle"], "slate", size=13)

s.region(24, 208, 832, 262, "ALL · map-reduce", "data", label_pos="tr")
s.arrow([(270, 160), (270, 186), (124, 186), (124, 238)], "data")
s.text(284, 178, "all: summarize, extract every clause", size=11.5, weight=600, fill="#0A5A51", anchor="start")
s.box(44, 240, 170, 84, "Chunk by section", ["clauses, with overlap;", "section title in each"], "data", size=13.5)
for y, n in [(250, "1"), (306, "2"), (362, "N")]:
    s.arrow([(214, 282), (242, 282), (242, y), (278, y)], "data")
    s.box(280, y - 23, 190, 46, f"Map: chunk {n}", ["records + source location"], "compute", size=12.5)
    s.arrow([(470, y), (528, 306 + (y - 306) * 0.3)], "compute")
s.box(530, 256, 180, 100, "Reduce", ["merge and dedupe records;", "reduce hierarchically if", "map outputs overflow"], "model", size=13.5)
s.arrow([(710, 306), (748, 306)], "output")
s.pill(802, 306, 100, 40, "Output", "output")

s.add('<line x1="124" y1="326" x2="124" y2="346" stroke="#D6408F" stroke-width="1.6" stroke-dasharray="4 3"/>')
s.box(44, 348, 172, 104, "Cross-chunk links", ["a definition in section 1", "can change a clause in", "section 9: prepend a", "glossary to every chunk"], "pink", size=13)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/01-llm-fundamentals/q48-long-documents.svg")
