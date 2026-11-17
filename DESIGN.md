---
name: Transformers and LLMs from Scratch
description: An interactive guide to Transformers and LLMs where the interface is graphite and white, and colour is kept for the data.
colors:
  ink: "#181c20"
  ink-hover: "#2e353b"
  muted: "#545c63"
  rule: "#dde1df"
  paper: "#f6f7f6"
  surface: "#ffffff"
  panel: "#eceeed"
  tint: "#e8ebe9"
  night-paper: "#15191b"
  night-surface: "#1d2225"
  night-panel: "#191d20"
  night-ink: "#e8ebea"
  night-muted: "#a3aba9"
  night-rule: "#30373a"
  night-tint: "#2a3134"
  query-blue: "#2a62c9"
  key-purple: "#7b3fb0"
  value-green: "#1a7a4b"
  residual-magenta: "#b02a7c"
  data-amber: "#925d00"
  data-teal: "#0b7480"
  warning-red: "#b42318"
  done-green: "#1f8a56"
  highlighter-yellow: "#ffe9a3"
  level-foundation: "#145c35"
  level-foundation-bg: "#d9efe2"
  level-core: "#1d4894"
  level-core-bg: "#dce7fa"
  level-deep: "#5c2a8c"
  level-deep-bg: "#ebdff7"
typography:
  display:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "clamp(2.25rem, 5vw, 3rem)"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "clamp(1.75rem, 3.6vw, 2.25rem)"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.015em"
  title:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "1.3rem"
    fontWeight: 700
    lineHeight: 1.15
  body:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.6
  label:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "0.88rem"
    fontWeight: 700
  mono:
    fontFamily: "JetBrains Mono, ui-monospace, Cascadia Code, Consolas, monospace"
    fontSize: "0.88em"
    fontWeight: 500
    lineHeight: 1.65
rounded:
  xs: "4px"
  sm: "6px"
  md: "8px"
  lg: "10px"
  pill: "999px"
spacing:
  xs: "6px"
  sm: "10px"
  md: "14px"
  lg: "18px"
  xl: "24px"
  section: "64px"
  gutter: "clamp(16px, 4vw, 56px)"
components:
  button:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "8px 14px"
    height: "38px"
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.sm}"
    padding: "8px 14px"
    height: "38px"
  button-outline-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.paper}"
  button-solid:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.surface}"
    rounded: "{rounded.sm}"
    padding: "8px 14px"
  button-solid-hover:
    backgroundColor: "{colors.ink-hover}"
  next-button:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "9px 18px"
  chip:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "8px 14px"
  chip-selected:
    backgroundColor: "{colors.tint}"
    textColor: "{colors.ink}"
  input:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "7px 10px"
    height: "38px"
  widget-card:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.lg}"
    padding: "18px clamp(12px, 2.5vw, 22px)"
  worked-panel:
    backgroundColor: "{colors.panel}"
    rounded: "{rounded.md}"
    padding: "14px 18px"
  quiz-option:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "8px 12px"
    height: "38px"
  level-pill-foundation:
    backgroundColor: "{colors.level-foundation-bg}"
    textColor: "{colors.level-foundation}"
    rounded: "{rounded.pill}"
    padding: "1px 9px"
  level-pill-core:
    backgroundColor: "{colors.level-core-bg}"
    textColor: "{colors.level-core}"
    rounded: "{rounded.pill}"
    padding: "1px 9px"
  level-pill-deep:
    backgroundColor: "{colors.level-deep-bg}"
    textColor: "{colors.level-deep}"
    rounded: "{rounded.pill}"
    padding: "1px 9px"
  settings-menu:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.lg}"
    padding: "0 8px 6px"
  mobile-nav-pill:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    height: "52px"
---

# Design System: Transformers and LLMs from Scratch

## Overview

**Creative North Star: "Colour is data"**

The interface is graphite on near-white (graphite on dark grey in the dark theme) and carries no brand colour at all. Every colour on the page is information: blue, purple and green are Query, Key and Value; the viridis ramp is the size of a number in a bar or a heatmap; green means done and red means wrong; the three level pills say how deep a section goes. When the reader sees colour, something is being measured.

The home page opens on the guide's running example, live: "The cat sat on the ___" with the probability of each next token drawn as bars coloured by their value, and buttons to sample a token and try another prompt. The caption says the scores are hand-written, because no trained model runs here.

The mood is calm and exact. One legible sans (Atkinson Hyperlegible Next, made for readers with low vision) carries headings, prose and controls; JetBrains Mono marks anything exact: every number a widget prints, shapes and code. The page is flat; only layers that float above it (menus, dialogs, sheets, the mobile pill, tooltips) cast a shadow.

**Key Characteristics:**
- Graphite and white chrome; links and main buttons are ink, never a hue.
- Colour only for data, state and level, each with exactly one meaning.
- One sans for words, one mono for numbers and code, both self-hosted.
- Flat content; shadows only on floating layers.
- Four box styles, never nested.
- Sentence case everywhere; no uppercase labels.
- Every text colour passes 4.5:1 on its background.

## Colors

A neutral grey scale for the interface and a small set of semantic hues that each mean exactly one thing.

### Interface (no hue)
- **Ink**: text, links (underlined at 45% strength, full and 2px on hover), outlined buttons, focus rings, the selected contents entry, the reading-progress bar, the on state of switches and checkboxes. In the code, `--accent` and `--fill` are both Ink.
- **Muted**: secondary text, labels, hints, icons in the Settings menu.
- **Rule**: every border and divider.
- **Paper** (the page), **Surface** (widgets, cards, menus, display equations), **Panel** (worked examples, inline code, bar tracks), **Tint** (the background of anything selected).
- **Night** variants: the same roles in the dark theme. The dark theme follows the system setting until the reader picks one in Settings.
- Code boxes are the one place colour is used for its own sake, so code reads like it does in an editor. The palette is Catppuccin: Latte on `#fafbfa` in the light theme (darkened where needed, hue and chroma kept, to reach 4.6:1) and Mocha on Night Surface in the dark theme. Mauve keywords, green strings, peach numbers, blue functions and calls, red builtins and italic `self`, yellow class names, teal keyword arguments, sky operators, grey italic comments.

### Data
- **Query Blue**, **Key Purple**, **Value Green**: the attention colours, for vectors, diagram borders, legend swatches and text tints.
- **Residual Magenta**: the residual stream in diagrams, drawn dashed.
- **Data Amber and Data Teal** (`--d1`, `--d2`; `#f0b849` and `#5fcfc4` in the dark theme): data that is not Q, K, V or a magnitude. Amber is the main or highlighted series (the solid line of a chart, the token being looked at, trained tokens, quantized values, generated text, the big model's tokens); teal is the comparison (the dashed line, the second class, the training data). Who-reads-whom grids use teal for source tokens and amber for target tokens. Signed values (LayerNorm's bars, the positional-encoding heatmap) use the blue–orange diverging scale `divColor()`. Never use `--accent` for data: it is the ink.
- **Viridis**: bar fills (`bars()` in `core.js`, the home demo) and heatmaps take their colour from their value through `viridis(t)`; a cut bar keeps its hatching instead.
- **Highlighter Yellow**: `mark`, the current row of a number table, the sampled token in the home demo.

### State
- **Done Green** (`--ok`): progress bars, ticks, the Done button when on, completed journey cells, correct quiz answers, a drill card marked "Got it".
- **Warning Red** (`--warn`): wrong answers, the "mistake" callout, the reset action, sync errors.

### Level
- **Level greens, blues and purples** (`level-*` with their `-bg` pairs): the filled Foundation, Core and Deep dive pills, each pair at least 6.6:1, tuned apart from Q/K/V. The only colour kept from the earlier design.

### Named Rules
**The Colour Is Data Rule.** Chrome never takes a hue. A new button, link, tab or highlight is Ink, Muted or a Tint; a hue appears only when it encodes data, a state or a level.

**The One Meaning Rule.** Blue, purple and green mean Query, Key and Value; Done Green and Warning Red mean done and wrong. New widgets reuse `--q`, `--k`, `--v`, `--ok`, `--warn` rather than inventing near-matches.

**The 4.5:1 Floor.** Every text colour passes 4.5:1 on the background it sits on, in both themes. When a borrowed palette fails, the colour is adjusted and the change is noted in a CSS comment.

## Typography

**Sans:** Atkinson Hyperlegible Next (variable, 200 to 800, with italics), self-hosted in `site/assets/fonts` under the OFL.
**Mono:** JetBrains Mono (variable, 100 to 800), self-hosted under the OFL.

**Character:** one sans designed for legibility carries the whole interface, so headings are distinguished by size and weight, not by a second family. Atkinson's only zero is slashed, so the zero alone comes from Hanken Grotesk: a family "Plain zero" holding just U+0030 (upright and italic, under 2 KB each) is listed first in `--sans`. Every other character, digits 1 to 9 included, stays Atkinson's. Code keeps JetBrains Mono's dotted zero. The mono marks anything exact and is never used for interface labels.

### Hierarchy
- **Display** (700, clamp(2.25rem, 5vw, 3rem), 1.15, −0.02em): page titles only.
- **Headline** (700, clamp(1.75rem, 3.6vw, 2.25rem), 1.15, −0.015em): the section title. The home demo's prompt sits between Display and Headline at clamp(1.55rem, 3.6vw, 2.1rem).
- **Title** (700, 1.3rem): subheadings inside a section.
- **Body** (400, 17px, 1.6; 16px at 900px and below): prose. The main column is capped at 820px, so lines stay under about 80 characters; the lede is 1.2rem and caps at 56ch.
- **Label** (700, 0.8–0.9rem): buttons, control labels, callout titles, pills, step counters. Counters use tabular numerals.
- **Mono** (500, 0.8–0.9rem): readouts, table cells, vectors, bar values, code boxes (0.8rem/1.65).

### Named Rules
**The Mono Means Exact Rule.** Mono is for numbers a widget computes, shapes and code. Labels, counters and buttons stay in the sans, with tabular numerals where digits must line up. Mono never uses ligatures, so `==` and `<=` show as typed.

**The Sentence Case Rule.** No `text-transform: uppercase` and no tracked-out caps labels.

**The Symbols Are Maths Rule.** Symbols such as d_model or √d_k are never plain text: KaTeX in static HTML, `sym()` in widget output. Tensor shapes stay in code style. Prose keeps 2px of room at the end of a line for the strut KaTeX puts after a subscript.

## Layout

A two-column reading layout on wide screens: a 270px sticky contents rail on the left and a main column up to 820px, inside a 1100px frame. Sections are generous vertical blocks with a rule between them, and each part page shows one view at a time. Side gutters scale with `clamp(16px, 4vw, 56px)`. Content is left-aligned throughout; only display equations are centred.

The home page is one column: title and lede, then the live next-token demo below a rule, then the Continue button and study time, the journey and a plain list of links. The journey is a list of parts split by rules, each with a 32px square letter (Ink border once started, filled green when complete) and its sections as a grid of rows, each led by a 10px cell that fills green when the section is done.

Breakpoints are 900px and 600px. At 900px and below the rail becomes a bottom sheet behind a scrim, a floating navigation pill appears, the body steps down to 16px and the title switches to its short form. At 600px and below the Settings popover becomes a bottom sheet and the drill bar drops the second half of its count.

Spacing follows a loose rhythm (6, 10, 14, 18, 24) inside components and opens up between them. Grids use `auto-fit` or `auto-fill` minimums so layouts reflow without breakpoints. Wide tables sit in a sideways-scrolling box with edge shadows; nothing widens the page.

## Elevation & Depth

Flat. Content surfaces are separated by a 1px Rule border and by tone (Paper, Surface, Panel), never by shadow. The top bar is solid Paper with one bottom rule.

### Shadow Vocabulary
- **Base lift** (`--shadow`: `0 1px 2px rgba(20,25,30,.06), 0 8px 24px rgba(20,25,30,.07)`; dark `0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25)`).
- **Popover** (`var(--shadow), 0 12px 32px rgba(0,0,0,.18)`): dropdown lists.
- **Menu** (`0 12px 32px rgba(0,0,0,.16), 0 1px 3px rgba(0,0,0,.08)`): the Settings popover. Dialogs use `var(--shadow), 0 16px 40px rgba(0,0,0,.3)`.
- **Floating pill** (`0 8px 28px rgba(0,0,0,.2)`): the mobile navigation pill, at 90% Surface with a 12px blur.
- **Sheet** (`0 -10px 40px rgba(0,0,0,.28)`): the contents bottom sheet on mobile.

### Named Rules
**The Float-Only Shadow Rule.** If it scrolls with the text, it has no shadow and no glow.

## Shapes

Small radii that grow with the size of the thing: 4px for inline marks, step counters, question numbers and bar fills; 6px for buttons, chips, inputs, quiz options, code blocks, steppers and the Done button; 8px for worked examples, display equations, quiz fieldsets, drill cards and dropdown lists; 10px for widget cards, the Settings menu and dialogs. Pills (999px) are kept for the level and optional pills and the mobile navigation bar.

Borders are 1px Rule by default; 1.5px where the border carries meaning (ink outlines, flow-diagram boxes). Notes use a 3px left rule instead of a box.

### Named Rules
**The Four Boxes Rule.** Four box styles only: card (widget, drill question), tinted panel (worked example, sample answer), left-rule note (callout, running example, open "why" note) and code block. Boxes are never nested.

## Components

### Buttons
- **Default:** Surface, Ink text at 700, 1px Rule border, 6px radius, at least 38px tall; hover turns the border to Ink. Large buttons are 46px with 11px × 20px padding.
- **Outline (main action):** transparent with a 1.5px Ink border; hover fills with Ink and the text turns to Paper. Used for Start, Continue, the next-part button, a widget's main action and the home demo's "Sample the next token".
- **Solid:** Ink with white text. One instance on the site (Connect in GitHub sync).
- **Danger:** Warning Red text and border; hover fills red.
- **Next button (section footer):** an Ink outline aligned right, showing only the next view's title.
- **Focus:** a 2px Ink outline, 2px offset, on every focusable element.
- Button text never ends in an arrow.

### Chips and steppers
Default button look in wrapping rows; selected chips take the Tint background, an Ink border and Ink text. Token chips stack the token over a small mono id.

### Cards / Containers
- **Widget card:** Surface, 1px Rule, 10px radius. Each opens with a muted line on what to look for and ends with a "What just happened?" note below a top rule.
- **Worked example:** Panel, no border, 8px radius; the exact numbers sit behind a disclosure.
- **Drill question:** a card at 8px radius with an Ink question number; "Got it" adds a 3px green left edge.
- **Callouts:** a 3px Ink left rule; only "mistake" uses Warning Red. The running example uses the same Ink rule.

### Bars
Value bars (`bars()`, the home demo) are a Panel track with a fill coloured by viridis from its value; the value sits at the right in mono with four decimals. The home demo's bars grow once on load and on a new prompt (0.6s), not under reduced motion.

### Inputs / Fields
Paper or Surface background, 1px Rule, 6px radius, 38px minimum. Ranges and checkboxes take Ink via `accent-color`. Every native select becomes a themed dropdown with the same field look; its list is a Surface popover at 8px radius.

### Navigation
- **Top bar:** solid Paper with one rule, 54px tall: a 28px logo (the four viridis squares) and the title on the left, a breadcrumb with a "3 of 6" counter on part pages, and an icon-only Settings button. A 2px Muted line along its bottom edge shows how far down the page you are; it is hidden on pages too short to need it.
- **Part letter:** a part is named by its letter in a square (2px Ink border), the same mark as on the home journey: beside the title on the part's opening screen and beside the part name at the top of the contents rail. There is no "Part A" label above a title.
- **Meta line:** under each section title, only the study time, the optional pill and the level pill. The part and the step are already in the top bar and the rail.
- **Part-complete screen:** headed "End of Part D: The wider picture", then one 14px cell per section (green when done) with the count, the unfinished sections, the next action and the key takeaways under a rule.
- **Contents rail:** muted entries at 0.92rem; the current entry is Ink at 700 on the Tint with a 3px Ink left rule; done entries get a green tick.
- **Mobile pill:** a 52px floating pill with previous and next arrows, an Ink step counter and the view title, and a progress line along its top edge. It hides while scrolling down.
- **Section footer:** previous link, the Done button (green when on), the bridge sentence and the next button.

### Settings Menu
A plain list on Surface at 10px radius: a left-aligned bold title over a rule, groups separated by rules, 42px rows with a muted 18px line icon, the label and a muted value with a chevron. Rows tint with Panel on hover. The Dark theme switch is a 38 × 22px track that turns Ink when on. Reset progress is red; actions in GitHub sync are underlined in bold. Sub-pages (Progress, GitHub sync) slide in with a back button; the Progress page shows each part as a lettered square, a green bar and "n of m". Below 600px it becomes a bottom sheet with a drag handle and a scrim. Motion is removed under reduced motion.

### Quiz
Options are full-width buttons on Paper at 6px radius. A right answer gets a green border and a 5px inset green bar; a wrong one the same in red. The score beside "Check yourself" is green.

### Pills
- **Level pill:** filled soft colour with matching dark text (Foundation green, Core blue, Deep dive purple).
- **Optional pill:** a grey mixed from Ink and Paper, at least 7.9:1, marking optional material and the "+N min optional" time.

### Code Box
A collapsed "In Python: …" disclosure; inside, a copy button and a 6px-radius code block at 0.8rem/1.65 in the theme's code palette.

## Do's and Don'ts

### Do:
- **Do** use `--q`, `--k`, `--v` for Query, Key and Value, `--res` (dashed) for the residual stream, `viridis()` for magnitudes, `divColor()` for signed values and `--d1` / `--d2` for any other main and comparison data.
- **Do** make a new widget's main action an Ink outline (`.controls button.pri`) and show selection with Tint, Ink border and Ink text.
- **Do** use `--ok` for anything done and `--warn` for anything wrong.
- **Do** print every number a widget computes in mono, to four decimals where it is a working value.
- **Do** keep every interactive control at least 38px tall (42–44px in Settings rows and the mobile pill).
- **Do** define every new colour as a variable on `:root` with a dark-theme override, and check 4.5:1 in both themes.
- **Do** turn off motion under `prefers-reduced-motion`.

### Don't:
- **Don't** give chrome (links, buttons, tabs, headers, icons) a hue.
- **Don't** add a second solid-filled button.
- **Don't** use mono for interface labels, or a second sans or a serif anywhere.
- **Don't** use `text-transform: uppercase` or tracked-out caps.
- **Don't** nest one box style inside another, or invent a fifth.
- **Don't** add shadows or glows to content that scrolls with the text.
- **Don't** load fonts or anything else from the network beyond KaTeX; the site must work offline from `file://`.
- **Don't** write symbols such as d_model as plain text.
- **Don't** use `alert` or `confirm`; use the themed modal dialog.
