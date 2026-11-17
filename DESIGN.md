---
name: Transformers and LLMs from Scratch
description: An interactive guide to Transformers and LLMs, read like an engineer's annotated notebook.
colors:
  lamp-amber: "#8a4b00"
  amber-fill: "#a35a00"
  amber-fill-hover: "#8a4b00"
  amber-tint: "#fbeed9"
  night-periwinkle: "#8ea5ff"
  periwinkle-fill: "#4c66e0"
  periwinkle-fill-hover: "#4259cf"
  periwinkle-tint: "#222838"
  query-blue: "#2a62c9"
  key-purple: "#7b3fb0"
  value-green: "#1a7a4b"
  residual-magenta: "#b02a7c"
  warning-red: "#b42318"
  success-green: "#1f8a56"
  highlighter-yellow: "#ffe7a0"
  paper: "#faf8f3"
  card-white: "#ffffff"
  linen-panel: "#f3efe6"
  ink: "#1c2230"
  slate-muted: "#5a6373"
  rule-line: "#e1dbce"
  night-page: "#13151a"
  night-card: "#1c1f26"
  night-panel: "#171a20"
  night-ink: "#e9ebef"
  night-muted: "#a2a7b1"
  night-line: "#30343d"
  level-foundation: "#145c35"
  level-foundation-bg: "#d9efe2"
  level-core: "#1d4894"
  level-core-bg: "#dce7fa"
  level-deep: "#5c2a8c"
  level-deep-bg: "#ebdff7"
typography:
  display:
    fontFamily: "Iowan Old Style, Palatino Linotype, Palatino, Georgia, serif"
    fontSize: "clamp(2rem, 5vw, 3.1rem)"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  headline:
    fontFamily: "Iowan Old Style, Palatino Linotype, Palatino, Georgia, serif"
    fontSize: "clamp(1.6rem, 3.6vw, 2.15rem)"
    fontWeight: 600
    lineHeight: 1.2
  title:
    fontFamily: "Iowan Old Style, Palatino Linotype, Palatino, Georgia, serif"
    fontSize: "1.28rem"
    fontWeight: 600
    lineHeight: 1.2
  body:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, sans-serif"
    fontSize: "17px"
    fontWeight: 400
    lineHeight: 1.65
  label:
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue, sans-serif"
    fontSize: "0.86rem"
    fontWeight: 600
  mono:
    fontFamily: "ui-monospace, Cascadia Code, Consolas, Menlo, monospace"
    fontSize: "0.88em"
    fontWeight: 500
    lineHeight: 1.6
rounded:
  hairline: "4px"
  sm: "8px"
  md: "10px"
  lg: "12px"
  xl: "14px"
  sheet: "18px"
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
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "9px"
    padding: "8px 14px"
    height: "38px"
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.lamp-amber}"
    typography: "{typography.label}"
    rounded: "9px"
    padding: "8px 14px"
    height: "38px"
  button-outline-hover:
    backgroundColor: "{colors.amber-tint}"
    textColor: "{colors.lamp-amber}"
  button-solid:
    backgroundColor: "{colors.amber-fill}"
    textColor: "{colors.card-white}"
    rounded: "9px"
    padding: "8px 14px"
  button-solid-hover:
    backgroundColor: "{colors.amber-fill-hover}"
  next-button:
    backgroundColor: "transparent"
    textColor: "{colors.lamp-amber}"
    rounded: "{rounded.md}"
    padding: "9px 18px"
  chip:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "9px"
    padding: "8px 14px"
  chip-selected:
    backgroundColor: "{colors.amber-tint}"
    textColor: "{colors.lamp-amber}"
  input:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "7px 10px"
    height: "38px"
  widget-card:
    backgroundColor: "{colors.card-white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.xl}"
    padding: "18px clamp(12px, 2.5vw, 22px)"
  worked-panel:
    backgroundColor: "{colors.linen-panel}"
    rounded: "{rounded.lg}"
    padding: "14px 18px"
  quiz-option:
    backgroundColor: "{colors.paper}"
    textColor: "{colors.ink}"
    rounded: "9px"
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
  mobile-nav-pill:
    backgroundColor: "{colors.card-white}"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    height: "52px"
---

# Design System: Transformers and LLMs from Scratch

## Overview

**Creative North Star: "The Annotated Notebook"**

The guide reads like a careful engineer's lab notebook: warm paper, serif headings set like chapter titles, margin notes in a thin coloured rule, and small live experiments worked out right on the page. The mood is calm, exact and warm. Prose is unhurried and set in a comfortable 17px sans; every number a widget prints is in monospace and shown to four decimals, because exactness is part of the voice. In the light theme the paper is cream and the accent is a lamp-lit amber; in the dark theme the page turns to a cool grey with only a hint of blue and the accent becomes a soft periwinkle.

The page itself stays flat and quiet. Depth appears only on things that float above the reading (menus, dialogs, the mobile navigation pill, tooltips) and as the soft colour halos behind the home title. Colour carries meaning, not decoration: blue, purple and green always mean Query, Key and Value; green, blue and purple pills mark Foundation, Core and Deep dive; the logo's four viridis squares echo the heatmaps inside the widgets.

Controls are refined and restrained: outlines rather than fills, tinted backgrounds for anything selected, and one solid button on the whole site. Settings borrows the iOS grouped list (coloured icon squares, small switches, sub-pages that slide in), which is the one place the system leans towards a native app.

**Key Characteristics:**
- Warm cream paper (light) and cool blue-grey (dark), each a full theme of its own.
- Serif headings, sans body, monospace for every number, shape and code.
- Flat content; shadows only on floating layers and the home halos.
- Colour is semantic: Q/K/V, levels, done (green), warning (red), accent for links and actions.
- Four box styles, never nested.
- Sentence case everywhere; no uppercase labels.
- Every text colour passes 4.5:1 on its background.

## Colors

A warm neutral notebook with a single accent per theme and a small set of semantic hues that each mean exactly one thing.

### Primary
- **Lamp Amber** (light theme accent): links, accent text, outlined buttons, the active contents entry, focus rings. Used sparingly so it always reads as "you can act here".
- **Amber Fill** (light theme solid fill): only the one solid button (Connect in GitHub sync) and the on state of switches. Hover deepens to Lamp Amber.
- **Amber Tint**: the background of anything selected (chips, steppers, the current contents entry, outlined-button hover).
- **Night Periwinkle** (dark theme accent), **Periwinkle Fill** and **Periwinkle Tint**: the same three roles in the dark theme.

### Secondary
- **Query Blue**, **Key Purple**, **Value Green**: the attention colours. Vectors, flow-diagram borders, legend swatches and text tints for Q, K and V. Value Green doubles as "done" (progress bars, ticks, correct quiz answers, the running-example rule).
- **Residual Magenta**: the residual stream in diagrams, drawn dashed.

### Tertiary
- **Level greens, blues and purples** (`level-*` with their `-bg` pairs): the filled Foundation, Core and Deep dive pills in the meta line, each pair at least 6.6:1. They are tuned apart from Q/K/V on purpose.
- **Highlighter Yellow**: `mark`, the current row of a number table, a highlighted bar.
- **Warning Red** and **Success Green**: wrong answers, the reset action, sync errors; the sync-ok dot.

### Neutral
- **Paper** (light page), **Card White** (widgets, cards, quiz fieldsets, display equations), **Linen Panel** (worked examples, readouts, inline code, captions).
- **Ink** (text), **Slate Muted** (secondary text, labels, hints), **Rule Line** (every border and divider).
- **Night Page**, **Night Card**, **Night Panel**, **Night Ink**, **Night Muted**, **Night Line**: the same roles in the dark theme. The dark theme follows the system setting until the reader picks one in Settings.
- Code boxes have their own palette: Gruvbox Light on a paler cream (`#fcf9f1`) in the light theme, Dracula's token colours on Night Card in the dark theme, comments and strings adjusted to pass 4.5:1.

### Named Rules
**The One Meaning Rule.** Blue, purple and green mean Query, Key and Value and nothing else; the accent is never used for any of them, and new widgets reuse `--q`, `--k`, `--v` rather than inventing near-matches.

**The One Solid Button Rule.** The whole site has a single solid-filled button. Start, Continue, a widget's main action and the footer's next button are accent outlines; selection is a tint, not a fill.

**The 4.5:1 Floor.** Every text colour passes 4.5:1 on the background it sits on, in both themes. When a borrowed palette (Gruvbox, Dracula) fails, the colour is adjusted, and the adjustment is noted in a CSS comment.

## Typography

**Display Font:** Iowan Old Style (with Palatino Linotype, Palatino, Georgia, serif)
**Body Font:** the system UI sans (system-ui, -apple-system, Segoe UI, Roboto, Helvetica Neue)
**Label/Mono Font:** ui-monospace (with Cascadia Code, Consolas, Menlo)

**Character:** a bookish serif for the headings that structure the reading, a neutral system sans that disappears under long study, and a monospace that marks anything exact: numbers, shapes, code, step counters. No web fonts are loaded; the stacks use what each platform already has.

### Hierarchy
- **Display** (600, clamp(2rem, 5vw, 3.1rem), 1.2, −0.01em): page titles in the hero only. Balanced wrapping.
- **Headline** (600, clamp(1.6rem, 3.6vw, 2.15rem), 1.2): one per section, the section title.
- **Title** (600, 1.28rem, 1.2): subheadings inside a section; wrap-up headings drop to 1.08rem so they sit a step below.
- **Body** (400, 17px, 1.65; 16px at 900px and below): prose. The main column is capped at 940px, so lines stay readable; the home lede caps at 60ch.
- **Label** (600, 0.84–0.88rem, sans): control labels, callout titles, button text, meta lines, pills.
- **Mono** (500, 0.8–0.9rem): readouts, table cells, vectors, code boxes (0.8rem/1.6), question numbers and the "3 of 6" step counters.

### Named Rules
**The Serif Speaks First Rule.** Serif is for headings and the brand only. Anything the reader touches or scans (buttons, labels, menus, pills) is sans; anything exact is mono.

**The Sentence Case Rule.** No `text-transform: uppercase` anywhere, and no tracked-out caps labels. Labels are written in sentence case.

**The Symbols Are Maths Rule.** Symbols such as d_model or √d_k are never plain text: KaTeX in static HTML, `sym()` in widget output. Tensor shapes stay in code style.

## Layout

A two-column reading layout on wide screens: a 270px sticky contents rail on the left and a main column up to 940px, inside a 1220px frame. Sections are generous vertical blocks (64px top padding, a rule between them), and each part page shows one view at a time. Side gutters scale with `clamp(16px, 4vw, 56px)`.

Breakpoints are 900px and 600px. At 900px and below the rail becomes a bottom sheet behind a scrim, a floating navigation pill appears 12px from the bottom edge, the body steps down to 16px and the title switches to its short form. At 600px and below the Settings popover becomes a bottom sheet and the drill bar drops the second half of its count.

Spacing follows a loose 4px-ish rhythm (6, 10, 14, 18, 24) inside components and opens up between them: 20–24px around widgets, callouts and worked examples, 40px before a section footer. Grids use `auto-fit` with 240–260px minimums so cards and two-column widget layouts reflow without breakpoints. Wide tables sit in a sideways-scrolling box with edge shadows; nothing widens the page.

## Elevation & Depth

Flat by default. Content surfaces are separated by a 1px Rule Line border and by tone (Paper, Card White, Linen Panel), never by shadow. Shadows appear only on layers that float above the reading, and on the home page's halos, which are blurred colour glows rather than drop shadows.

### Shadow Vocabulary
- **Base lift** (`box-shadow: 0 1px 2px rgba(20,25,40,.06), 0 8px 24px rgba(20,25,40,.06)`; dark `0 1px 2px rgba(0,0,0,.4), 0 8px 24px rgba(0,0,0,.25)`): the `--shadow` token, combined with a larger drop for floating layers.
- **Popover** (`var(--shadow), 0 12px 32px rgba(0,0,0,.18)`): dropdown lists.
- **Menu** (`var(--shadow), 0 18px 48px rgba(0,0,0,.24)`): the Settings popover; dialogs use `0 16px 40px rgba(0,0,0,.3)`.
- **Floating pill** (`0 8px 28px rgba(0,0,0,.2)`): the mobile navigation pill, frosted at 90% surface with a 12px blur.
- **Sheet** (`0 -10px 40px rgba(0,0,0,.28)`): the contents bottom sheet on mobile.
- **Accent glow** (`0 8px 28px color-mix(in srgb, var(--accent) 16%, transparent)`): home cards on hover, the Continue button (22%, 32% on hover), and 5px rings around started and completed parts on the journey map. Home page only.

The top bar is frosted glass: 62% page colour with an 18px blur and 170% saturation, so content scrolls under it softly.

### Named Rules
**The Float-Only Shadow Rule.** If it scrolls with the text, it has no shadow. Shadows mark what floats (menus, dialogs, sheets, the mobile pill, tooltips); the glow vocabulary belongs to the home page.

## Shapes

Gently rounded, never sharp and never blobby. Radius grows with the size of the thing: 4–5px for inline code, vector cells and heatmap cells; 8–9px for inputs, buttons, chips, quiz options and code boxes; 10–12px for worked examples, display equations, quiz fieldsets and drill cards; 14px for widget cards, home cards and dialogs; 16px for the Settings popover and 18px for bottom sheets. Pills (999px) are reserved for small status and navigation pieces: level and optional pills, steppers, the Done button, step counters and the mobile navigation bar.

Borders are 1px Rule Line by default; 1.5px is used where a border carries meaning (accent outlines, flow-diagram boxes, badges). Notes use a 3px left rule in their meaning colour instead of a box.

### Named Rules
**The Four Boxes Rule.** There are four box styles only: card (widget, drill question), tinted panel (worked example, sample answer), left-rule note (callout, running example, open "why" note) and code block. Boxes are never nested inside one another.

## Components

### Buttons
Quiet by default, accent when they lead somewhere.
- **Shape:** gently rounded (9px), at least 38px tall; large buttons 46px with 11px × 20px padding.
- **Default:** Paper background, Ink text, 1px Rule Line border; hover turns the border to the accent.
- **Outline (main action):** transparent with a 1.5px accent border and accent text; hover fills with the accent tint. Used for Start, Continue, the next-part button and a widget's main action.
- **Solid:** Amber Fill with white text, hover one step deeper. One instance on the site (Connect in GitHub sync).
- **Danger:** Warning Red text and border; hover fills red with surface-coloured text.
- **Next button (section footer):** an accent outline at 10px radius and 9px × 18px padding, aligned right, showing only the next view's title.
- **Focus:** a 3px accent outline, 2px offset, on every focusable element.
- **Disabled:** 45% opacity.

### Chips
- **Style:** the default button look, used in wrapping rows with a 6px gap; token chips stack the token over a small mono id.
- **Selected:** accent tint background, accent border and accent text. Steppers use the same treatment in pill form, with completed steps keeping only the accent border.

### Cards / Containers
- **Widget card:** Card White, 1px Rule Line, 14px radius, 18px vertical and fluid horizontal padding. Each opens with a muted line on what to look for and ends with a "What just happened?" note below a top rule.
- **Worked example:** Linen Panel, no border, 12px radius, 14px × 18px padding; the exact numbers sit behind a "Work through the exact numbers" disclosure.
- **Drill question:** a card at 12px radius; a correct mark adds a 3px Value Green left edge.
- **Home cards:** 14px radius, accent border and an accent glow on hover.

### Inputs / Fields
- **Style:** Paper background, 1px Rule Line, 8px radius, 38px minimum height, 0.95rem text. Ranges and checkboxes take the accent via `accent-color`.
- **Dropdowns:** every native select is replaced by a themed dropdown with the same field look and a drawn chevron that flips when open; the list is a Card White popover at 10px radius, the active option tinted and the selected one ticked in the accent.
- **Focus:** the border turns to the accent on hover and while open; keyboard focus draws the 3px accent ring.

### Navigation
- **Top bar:** frosted glass, 54px tall, logo and serif title on the left, a breadcrumb with a mono "3 of 6" pill on part pages, and an icon-only Settings button.
- **Contents rail:** muted entries at 0.92rem with 8px radius; hover takes the Linen Panel tint; the current entry is Ink, semibold, on the accent tint with a 3px accent left rule; done entries get a green tick.
- **Mobile pill:** a frosted 52px pill floating 12px above the bottom edge, with previous and next arrows, a mono step pill and the view title in the middle, and a 3px progress line along its top edge. It hides while scrolling down.
- **Section footer:** previous link, a pill-shaped Done button that turns green when on, the bridge sentence and the next button.

### Settings Menu
The iOS-style grouped list: rounded groups (12px) of 44px rows on the page colour, each row led by a white line icon on a 24px coloured square (purple theme, green progress, grey sync, red reset). Switches are 38 × 22px and fill with the accent when on. Sub-pages slide in from the right with a back button; there is no close button. Below 600px it becomes a bottom sheet with a drag handle and a 50% black scrim. All motion is removed under reduced motion.

### Quiz
Options are full-width buttons on Paper at 9px radius. A right answer gets a Value Green border and a 5px inset green bar on the left; a wrong one the same in Warning Red. The score sits beside the "Check yourself" heading in green.

### Pills
- **Level pill:** filled soft colour with matching dark text (Foundation green, Core blue, Deep dive purple), 0.76rem semibold, in the meta line under each section title.
- **Optional pill:** a grey fill mixed from Ink and the page colour, 0.72rem semibold, at least 7.9:1, marking optional headings, boxes and the "+N min optional" time.

### Code Box
A collapsed "In Python: …" disclosure with a rotating chevron; inside, a copy button and an 8px-radius code block at 0.8rem/1.6, highlighted with the theme's code palette.

## Do's and Don'ts

### Do:
- **Do** use `--q`, `--k`, `--v` for Query, Key and Value, and `--res` (dashed) for the residual stream, in every new widget.
- **Do** make the main action of a new widget an accent outline (`.controls button.pri`), and show selection with the accent tint, accent border and accent text.
- **Do** put shadows only on floating layers (menus, dialogs, sheets, the mobile pill, tooltips).
- **Do** keep every interactive control at least 38px tall (44px in the mobile pill and Settings rows).
- **Do** print every number a widget computes in monospace, to four decimals where it is a working value.
- **Do** define every new colour as a variable on `:root` with a dark-theme override, and check it reaches 4.5:1 in both themes.
- **Do** set new wide tables inside the sideways-scrolling box, with `--sbg` matching the background behind it.
- **Do** turn off motion under `prefers-reduced-motion`.

### Don't:
- **Don't** add a second solid-filled button.
- **Don't** use `text-transform: uppercase` or tracked-out caps labels.
- **Don't** nest one box style inside another, or invent a fifth box style.
- **Don't** use the accent, or a near-match of blue, purple or green, for anything that is not its fixed meaning.
- **Don't** add shadows to content that scrolls with the text, or bring the home glows onto other pages.
- **Don't** load web fonts or any network resource beyond KaTeX; the site must work offline from `file://`.
- **Don't** write symbols such as d_model as plain text.
- **Don't** use `alert` or `confirm`; use the themed modal dialog.
