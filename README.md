# Transformers and LLMs from Scratch

An interactive visual guide to how GPT-style language models work, from tokens to training to serving, plus an interview question bank for AI engineers.

**Live site:** https://azaharuddinruhit.github.io/transformers-from-scratch/ (once GitHub Pages is enabled, see [Deployment](#deployment))

## What's inside

| Part | Sections | What you learn |
| --- | --- | --- |
| A · The big picture | 1–2 | What an LLM computes, and why Transformers replaced recurrent networks |
| B · Building the model | 3–8 | Tokens, positions, self-attention, multi-head attention, the Transformer block, the full model |
| C · Training and using it | 9–12 | Cross-entropy training, decoding, KV cache and serving, scaling laws |
| D · The wider picture | 13–14 | Encoder–decoder models and the 2017 paper in detail |
| E · Interview prep | 15–19 | About 60 interview questions with bullet-point answers: fundamentals, RAG, fine-tuning, agents, production |
| Reference | 20 | Glossary and further reading |

Each section in Parts A to D has learning objectives, a running example, equations built up step by step with every symbol named, a worked example you can check by hand, interactive widgets, key takeaways, a self-check quiz and discussion questions.

Highlights:

- A tiny Transformer (about 9,600 parameters) that trains in your browser, plus a generation playground with temperature, top-k and top-p.
- Step-by-step attention, KV-cache, parameter-count and inference-cost calculators.
- A settings menu in the header with "Exact numbers" or "Simplified" detail, a Light, Dark or Auto (system) theme, an "Essentials only" switch for a first pass, and a progress reset.
- A "Mark this section as done" button after every section. Progress (sections done and a percentage weighted by study time) shows in the contents panel together with the study time left, with ticks next to finished sections, and is saved in your browser only, so it does not sync between devices. A "Reset progress" button clears it.
- On phones, a slim floating bar with previous and next buttons, the current section and a thin progress line. It hides while you scroll down to read and comes back when you scroll up; tapping it opens the contents as a bottom sheet.
- Themed scrollbars and dropdown menus that follow the light or dark theme.
- A five-session plan for teaching the material to a team.

All numbers are computed in the browser. No pretrained model is used; where a demonstration is simplified (hand-picked attention weights, illustrative scaling curves), the page says so.

## Using it

Open `transformers-from-scratch.html` in any modern browser. It is a single self-contained file. The only network request loads the KaTeX maths renderer from a CDN; offline, equations show as LaTeX source and everything else still works.

## Deployment

`.github/workflows/pages.yml` publishes the guide on every push to `main` or `master`, serving the HTML file as the site's `index.html`.

One-time setup: in the repository, go to **Settings → Pages → Build and deployment** and set **Source** to **GitHub Actions**.

## Caveats

- Facts about the 2017 paper (BLEU scores, model sizes, training times) were written from memory; verify them against the paper before relying on them.
- Tested in Chrome on desktop and at phone width. Not yet tested on Safari, Firefox or with a screen reader.
