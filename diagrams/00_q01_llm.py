"""Must Know Q1: what an LLM is: next-token prediction run in a loop."""
import sys; sys.path.insert(0, __file__.rsplit("/diagrams/", 1)[0] + "/scripts")
from svg import Svg

s = Svg(860, 404, "An LLM predicts the next token, in a loop", "Layers score every vocabulary token, one is sampled and appended, and the whole pass runs again.")
y, h = 92, 76
s.pill(85, y + h / 2, 110, 44, "Prompt", "human")
s.arrow([(140, 130), (160, 130)], "human")
s.box(164, y, 140, h, "Tokenizer", ["text → subword", "tokens → vectors"], "data")
s.arrow([(304, 130), (324, 130)], "data")
s.box(328, y, 160, h, "Transformer", ["attention and", "feed-forward layers"], "model")
s.arrow([(488, 130), (508, 130)], "model")
s.box(512, y, 140, h, "Logits", ["one score per", "vocabulary token"], "compute")
s.arrow([(652, 130), (672, 130)], "compute")
s.box(676, y, 160, h, "Sample a token", ["softmax gives", "probabilities; pick one"], "output")
s.arrow([(756, 168), (756, 208), (408, 208), (408, 172)], "output")
s.text(582, 224, "append the token, repeat", size=11.5, weight=600, fill="#275C1C")

# how a base model becomes an assistant
s.region(24, 250, 410, 134, "Two training stages", "data")
s.box(44, 284, 160, 76, "Pretraining", ["next-token on text", "→ base model"], "data", size=13.5)
s.arrow([(204, 322), (230, 322)], "data")
s.box(234, 284, 180, 76, "Post-training", ["instruction + preference", "tuning → assistant"], "model", size=13.5)

# what it does not have
s.region(450, 250, 386, 134, "What it lacks", "fail")
for i, t in enumerate(["no memory between calls", "no knowledge past cutoff", "no notion of truth"]):
    s.pill(572, 298 + i * 32, 212, 26, t, "fail", size=11.5, weight=600)
s.arrow([(682, 330), (704, 330)], "output")
s.box(708, 298, 112, 64, "Covered by", ["RAG, tools", "and evals"], "output", size=12.5)
s.save(__file__.rsplit("/diagrams/", 1)[0] + "/assets/00-must-know/q01-llm.svg")
