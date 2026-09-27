# How every answer is written

The reader is smart but new to the topic. They may never have seen the term in the question. An answer is
good only if that reader can follow it from the first line to the last without looking anything up.

Think each question through before writing: what does a newcomer need to know first, in what order, and what
picture or number will make it click?

## The shape of an answer

1. **The answer in plain English.** One or two bold sentences. No jargon that has not been explained yet.
2. **The idea.** An intuition, an everyday analogy, or a small concrete example with real numbers
   ("say the model has seen the words *the cat sat on the*…"). This is where the reader "gets it".
3. **How it works.** Numbered steps or short bullets, in the order things actually happen.
   - Define every term the first time it appears, in a few words: "a token (a word or a piece of a word)",
     "the KV cache (a store of the keys and values already computed for earlier tokens)".
   - Spell out every acronym once: "retrieval-augmented generation (RAG)".
   - Never refer to something that has not been introduced.
4. **The formula, only if it helps.** Introduce it after the idea, never first: "Put as a formula…". Then read
   every symbol out in words, then work a tiny example with small numbers. Explain the maths vocabulary too:
   what log does and why it is there ("log turns multiplying many small probabilities into adding them, which
   is easier to compute"), what Σ means ("add up over every token"), what softmax does ("turns any scores into
   probabilities that add up to 1"). A formula dropped in with no explanation is a failure.
5. **Read the figure.** If the answer has an image, tell the reader what to look at, using the figure's real
   labels and colours: "In the figure, follow the green lane: the query is embedded, then…". Open the rendered
   figure (see below) so the labels you quote are the ones on it. Put a one-line italic caption directly under
   each image: `*Figure: what it shows, in one sentence.*`
6. **Watch out.** One line, labelled, with the common mistake, trade-off or production gotcha.

## Rules

- **Correct above all.** If unsure of a fact, leave it out or say it is approximate. Say whether a number is a
  typical range, a published result or a rule of thumb. Anything that changes fast (model names, prices,
  versions, laws) is dated "as of 2025–26".
- **Length:** about 200–350 words of prose per answer (system design up to ~420), not counting code, tables,
  formulas and images. Short paragraphs and bullets; never a solid block. No filler, no "Great question",
  no recap at the end.
- **Keep** the question heading exactly as it is, the numbering, the table of contents, every existing image tag,
  and every code block (you may tidy code comments). Keep tables if they help; explain what they compare.
- **No links out.** The answer must stand on its own. No "further reading", no references to other courses,
  sites or authors.
- **Math syntax for GitHub:** inline math is `$`…`$` (dollar, backtick … backtick, dollar); display math is a
  fenced block starting with ```math. Inside math never type `<` or `>`: write `\lt` and `\gt`
  (GitHub's HTML sanitiser swallows `<` and breaks the formula). Write money as "USD 5", never with a dollar sign.
- American spelling. No emoji.

## Seeing a figure

Every `<img src="../assets/...svg">` has an SVG file. Render it to look at it:

```
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --hide-scrollbars \
  --force-device-scale-factor=1.5 --window-size=W,H --screenshot=<your scratch dir>/fig.png file://<absolute svg path>
```

(W and H are the width and height attributes at the top of the SVG.) Then read the PNG.
