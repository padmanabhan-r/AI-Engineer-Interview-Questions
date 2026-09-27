# Brief: redraw Mermaid diagrams as hand-built SVGs

Diagrams are colourful, hand-built SVGs made with the kit in
`scripts/svg.py`. The user rejected Mermaid's plain boxes. The approved style is in four samples — study them first:

- `diagrams/01_q06_forward_pass.py` → `assets/01-llm-fundamentals/q06-forward-pass.svg` (a rail, branches, a mini grid)
- `diagrams/03_q02_rag_architecture.py` → `assets/03-retrieval-augmented-generation-rag/q02-basic-rag.svg` (two lanes, snake layout)
- `diagrams/04_q05_react.py` → `assets/04-ai-agents-and-agentic-systems/q05-react.svg` (a loop + a colour-coded side panel)
- Read `scripts/svg.py` fully: pill, box, cylinder, hexagon, diamond, add_node, region, arrow (straight, elbow, curve),
  rail, grid, text. Colour = role (model violet, data teal, compute blue, amber, output green, human navy, fail red,
  slate, pink). Shape = meaning (pill a step, box a component, cylinder stored data, hexagon a gate/check,
  diamond a decision).

## What you do, per diagram

1. Read the whole answer the Mermaid block sits in, not just the block. The SVG must show what the answer says,
   and may show it better (for example a small grid, a trace panel, tensor shapes, a before/after), but must not
   add facts, numbers or components the answer does not contain.
2. Write `diagrams/<NN>_q<QQ>[_<k>]_<slug>.py` (NN = topic number, QQ = question number, zero-padded; add `_2` if a
   question has a second diagram) that saves `assets/<topic-file-stem>/q<QQ>[-<k>]-<slug>.svg`. Copy the sample's
   two-line header for imports and the save path.
3. Give it a title (the idea in a few words) and a one-line subtitle (the takeaway). Width 760–900 px; height as
   needed. Font sizes: titles in boxes 13–14, detail 11–12. At most ~12 shapes. Leave air between things.
4. Run it. **Fix every "text may overflow" warning** (widen the shape, shorten the label, or split a line with \n).
5. Render and LOOK at it:
   `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 --window-size=W,H --screenshot=<your scratch dir>/x.png file://<abs path to svg>`
   then Read the PNG. Check: no text over a line or another shape, no arrow crossing a box, arrows point the way the
   flow goes, labels sit next to what they label, nothing clipped at the card edge. Fix and re-render until clean.
   This visual check is the whole point; do not skip it.
6. Append one line to your manifest `diagrams/manifest/<your-id>.tsv` (tab-separated, no header):
   `topics/<file>.md<TAB><question number><TAB><block index within that question, 1-based><TAB>assets/...svg<TAB><alt text, one sentence>`

## Rules

- **Do not edit any `topics/*.md` file**, `scripts/svg.py`, or anything another writer owns. The swap into the
  markdown is done centrally from the manifests. If the kit lacks something, draw it inline in your own script with
  `s.add('<svg element>')`.
- Keep all scratch files (PNGs) in your own folder under the scratchpad, named for your id. Never write to the
  scratchpad root.
- Do not run git.
- When done, report: diagrams made (must equal the Mermaid blocks in your share), the manifest path, and any
  diagram you simplified or could not make clean.
