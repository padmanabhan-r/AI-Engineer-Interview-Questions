// Render every Vega-Lite spec in charts/*.vl.json to assets/<topic>/<name>.svg, in the house style.
//   node scripts/vl2svg.mjs            (all)     node scripts/vl2svg.mjs charts/x.vl.json   (one)
import { readFileSync, writeFileSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, basename, join } from "node:path";
import * as vega from "vega";
import * as vl from "vega-lite";

const ROOT = new URL("..", import.meta.url).pathname;
const SANS = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif";
// The same palette as scripts/svg.py: model, data, compute, amber, output, pink, slate.
const RANGE = ["#6D4AE0", "#0E8C7E", "#2F6FE4", "#C98A06", "#3E8E2F", "#D6408F", "#667085"];
const CONFIG = {
  background: "#FFFFFF", padding: 20, font: SANS,
  view: { stroke: null },
  title: { fontSize: 18, fontWeight: 700, color: "#151A23", anchor: "start", subtitleColor: "#667085", subtitleFontSize: 12.5, subtitlePadding: 6, offset: 16 },
  axis: { labelColor: "#475467", titleColor: "#344054", labelFontSize: 11.5, titleFontSize: 12.5, titleFontWeight: 600, gridColor: "#EEF0F4", domainColor: "#D0D5DD", tickColor: "#D0D5DD" },
  legend: { labelColor: "#344054", titleColor: "#344054", labelFontSize: 12, titleFontSize: 12.5, orient: "top", direction: "horizontal" },
  range: { category: RANGE },
  bar: { cornerRadiusTopLeft: 4, cornerRadiusTopRight: 4 },
  line: { strokeWidth: 3 },
};

const files = process.argv.slice(2).length ? process.argv.slice(2) : readdirSync(join(ROOT, "charts")).filter((f) => f.endsWith(".vl.json")).map((f) => join("charts", f));
for (const f of files) {
  const { out, ...spec } = JSON.parse(readFileSync(join(ROOT, f), "utf8"));
  const compiled = vl.compile({ ...spec, config: { ...CONFIG, ...(spec.config ?? {}) } }).spec;
  const view = new vega.View(vega.parse(compiled), { renderer: "none" });
  let svg = await view.toSVG();
  // A rounded card, like the diagrams, so charts sit on the page the same way.
  svg = svg.replace(/<rect width="(\d+)" height="(\d+)" fill="#FFFFFF"(\/?)>/, '<rect width="$1" height="$2" rx="18" fill="#FFFFFF" stroke="#E4E7EC" stroke-width="1.5"$3>');
  const dest = join(ROOT, out);
  mkdirSync(dirname(dest), { recursive: true });
  writeFileSync(dest, svg);
  console.log(`${basename(f)} -> ${out}`);
}
