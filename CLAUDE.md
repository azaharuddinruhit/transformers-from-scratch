# Transformers and LLMs from Scratch

A single-file interactive guide used to learn Transformers and LLMs, train an engineering team, and prepare for AI engineer interviews. Published with GitHub Pages.

## Files

- `transformers-from-scratch.html`: the whole site. Vanilla HTML, CSS and JS in one file; no build step is needed to view or deploy it.
- `.github/workflows/pages.yml`: on push to `main` or `master`, copies the HTML to `_site/index.html` and deploys it to GitHub Pages. Repository setting required: Settings → Pages → Source: "GitHub Actions".

The page was generated from build sources (`head.html`, `bodyA.html`, `bodyB.html`, `core.js`, `ui.js`, patch scripts and a headless-Chrome test harness) kept in a temporary folder outside this repo. They may no longer exist. If they are absent, edit the HTML directly.

## Audience and purpose

- Readers: software engineers with some ML background who have not studied Transformers.
- Uses: self-study, live team sessions (a five-session plan sits in the "How to use this guide" box), and interview drilling (Part E).
- Rules from the original spec: never fabricate pretrained weights or pretend to run a real large LLM; whenever a simulation is simplified, say exactly what is simplified.

## Page structure

Twenty `<section class="ch" id="…" data-title="…">` elements in five parts plus a reference section:

- **Part A, the big picture**: `intro`, `why`
- **Part B, building the model**: `tokens`, `position`, `attention`, `multihead`, `block`, `architecture`
- **Part C, training and using it**: `training`, `generation`, `inference`, `scaling`
- **Part D, the wider picture**: `encdec`, `paper`
- **Part E, interview prep**: `iv-core`, `iv-rag`, `iv-ft`, `iv-agents`, `iv-prod`
- **Reference**: `glossary`

Each section in Parts A to D contains, in order: `div.num` label, `h2`, `nav.map` (the "you are here" strip), `div.start` (before / after / running example), intuition, `.flow` diagram, equations, `.worked` example (`.x-only` exact and `.s-only` simple variants), `.widget` interactives (each with a `p.look` line and a `.wjh` "What just happened?"), callouts, misconceptions, then `div.wrapup` (key takeaways, quiz `fieldset`s with `button.qo` and `data-ok` on the right answer, discussion questions, `p.bridge`).

Part E sections use `div.qa` blocks: `h4.q` with a `span.qn` number, a bullet `ul`, and an optional `p.fu` ("Likely follow-up" or "Trap", plus "Review" links).

## Conventions

- **Cross-references** are `<a class="xref" data-ref="sectionId" [data-to="sectionId"]></a>`; `initGuide()` fills in "Section n" text at load time. Never hard-code section numbers in prose.
- **Hard-coded numbering** that must change if sections are added or moved: every `div.num` label, every `nav.map` strip, and `GPARTS` in the script (first section id of each part → part name, used for TOC headings).
- **Notation**: one symbol per meaning, defined in the notation table at the top. Vectors are rows (`xW`). Use `T` for sequence length, `τ` for temperature, `L` for layers, `N` for parameter count, `𝓛` for loss, `P` for prompt length, `n_vocab`, `d_model`, `d_k`, `d_ff`, `E` / `E_pos` for embedding tables, `LayerNorm` (never `LN`). Shapes are written `[B, T, d_model]`.
- **Equation blocks**: question → plain-word reasoning → equation → `p.sym` naming every symbol → optional `div.why` box.
- **Maths**: KaTeX 0.16.11 from jsdelivr, the page's only network request. `<span class="tex">` is inline, `<div class="tex d">` is display. Escape `&` as `&amp;` and use `\lt` / `\gt`.
- **Themed controls**: scrollbars use `scrollbar-color` with theme variables. Every native `<select>` is hidden and replaced at load by an accessible custom dropdown (`initSelects()`, classes `.dd`, `.dd-btn`, `.dd-list`); the native element stays in the DOM as the source of truth, so keep reading `.value` and listening for `change` on it. Options must exist before `initSelects()` runs.
- **Header**: logo (inline SVG matching the favicon) and title on the left (`.lgt` long title on desktop, `.sm` short title at 900px and below), and a Settings button (`#setBtn`) on the right that opens the `#setMenu` dropdown. The menu holds the detail mode (`#modeExact`, `#modeSimple`), the theme (`.thBtn` with `data-t` light, dark or auto; auto removes `data-theme` and the `tfs-theme` key), an `.essBtn`, and `#pgReset2`, which forwards to `#pgReset`.
- **Progress**: `initChrome()` appends a `.donebar` with a `.doneBtn` to every section (before `nav.secnav`) and stores completed section ids as JSON in `localStorage` under `tfs-done` (wrapped in try/catch; the page works without storage). Progress is painted into the contents list, the map strips and the mobile bar by `paintDone()`. Section ids in the saved list are matched against current ids, so renaming an id resets that section.
- **Mobile navigation** (max-width 900px): there is no hamburger. `#mbar` is a floating pill (previous, `#mNum` + `#mTxt` for the current section, next, with a progress line `#mFill` on its top edge). It gets class `hid` while scrolling down and is shown again on scroll up, near the top or end of the page, while a section's Done bar is visible, while the sheet is open or the bar has focus, and for 0.9 s after a prev/next tap; tapping its middle button opens `#toc` as a bottom sheet with `#scrim`. Keep the mobile `#mbar{display:grid}` rule last in the stylesheet, because the base `display:none` rule would otherwise override it.
- **Optional content**: `.opt` elements hide when "Essentials only" is on (`body.ess`).
- **Style**: plain British-leaning English, short sentences, no marketing tone. Colours come from CSS variables on `:root`, with dark-mode overrides.

## Checking changes

- Open the HTML in a browser; the console should be free of errors and no equation should show raw LaTeX.
- Check light and dark themes, both Exact and Simple modes, Essentials only, and a 390 px wide viewport (no sideways scrolling).
- The toy model in Section 9 should train in the browser to a validation loss near 0.3 in about 600 steps.
- Paper facts in Section 14 (BLEU scores, model sizes, training times) were written from memory; verify them against the paper before presenting.
