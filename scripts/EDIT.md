# The editor pass: read as a newcomer, fix in place

You are the second pair of eyes. The reader is smart but new to AI engineering and to statistics: they know
basic school maths and what a program is, nothing more. Read every answer in your file slowly, as that reader,
and fix every place they would get stuck. Edit the file directly. Follow `scripts/STYLE.md` for everything else.

Go answer by answer. For each one, check and fix:

1. **Undefined terms.** Every technical term must be explained in a few plain words the first time it appears in
   that answer, even if an earlier answer explained it (readers jump straight to one question). Watch
   especially for: standard error, confidence interval, variance, distribution, sample, bootstrap, resampling,
   significance, p-value, chi-squared, log, logit, softmax, gradient, loss, embedding, vector, token, parameter,
   weights, layer, attention, latency, throughput, GPU memory, quantization, fine-tuning, inference.
2. **Undefined symbols.** Every letter in a formula is said in words right next to it ("where $`p`$ is the pass
   rate and $`n`$ the number of test items"). If a formula has no plain-words reading and no small worked
   example, add them. If the formula does not help a newcomer, replace it with words.
3. **Jumps.** A step that assumes something not yet said: add the missing sentence. Keep the order in which
   things actually happen.
4. **The figure.** If the answer has an image, render it (see STYLE.md), look at it, and make sure the text tells
   the reader what to look at using labels that really appear in it, and that there is a one-line
   `*Figure: …*` caption under the image. Fix any mismatch between text and figure.
5. **Correctness.** Fix anything wrong. Don't add facts you are unsure of.
6. **Length.** Stay within about 200–350 words of prose (system design up to ~420). Make room by cutting
   repetition, not explanations.

Never change question headings, numbering, the table of contents, image tags or code (beyond comments). Math:
inline `$`…`$`, display ```math, and never a bare `<` or `>` inside math (use `\lt`, `\gt`).

When done run `python3 scripts/check.py` and `node scripts/check-math.mjs`, and report: how many answers you
changed, the most common kind of gap you fixed, and anything you were unsure about.
