# Transformers and LLMs from Scratch

An interactive visual guide to how GPT-style language models work, from tokens to training to running models yourself, plus an interview drill for AI engineers.

**Live site:** https://azaharuddinruhit.github.io/transformers-from-scratch/

## What's inside

| Page | Sections | What you learn |
| --- | --- | --- |
| Part A · The big picture | 2 | What an LLM computes, and why Transformers replaced recurrent networks |
| Part B · Building the model | 6 | Tokens, positions, self-attention, multi-head attention, the Transformer block, the full model |
| Part C · Training and using it | 7 | Cross-entropy training, scaling laws, post-training (SFT, RLHF and DPO, reasoning models), decoding, GPUs and CUDA, inference in practice, open-weight models with Ollama and vLLM |
| Part D · The wider picture | 3 | Encoder–decoder models, the 2017 paper in detail, and what modern LLMs change (RoPE, RMSNorm, SwiGLU, GQA, mixture of experts) |
| Interview drill | 6 topics | 65 questions, each with a short sample answer and bullet points: fundamentals, RAG, fine-tuning, agents, production, hardware |
| Reference | 3 | Glossary, notation and further learning |

Each section in Parts A to D has learning objectives, a running example, equations built up step by step with every symbol named, a worked example you can check by hand, interactive widgets, key takeaways, a self-check quiz and discussion questions.

Highlights:

- Two paths: **Full depth** (about 4 h 25 min) and **Quick tour** (about 3 h 50 min, simplified worked examples, optional material hidden). You can switch at any time.
- A tiny Transformer (about 9,600 parameters) that trains in your browser, plus a generation playground with temperature, top-k and top-p.
- Step-by-step attention, KV-cache, GPU-memory, parameter-count and inference-cost calculators.
- "In Python" boxes with runnable NumPy and PyTorch code and its real output, collapsed by default.
- The interview drill hides each answer behind "Show answer"; mark questions "Got it" or "Review again" and filter by topic or status.
- Progress (sections done, quiz answers, drill marks, where you left off) is saved in your browser only, so it does not sync between devices. Settings has a progress reset, the path, a Light, Dark or Auto theme, and whether code boxes start expanded.
- On phones, a floating bar with previous and next buttons, the current section and a progress line; tapping it opens the contents as a bottom sheet.
- A six-session plan on the home page for teaching the material to a team.

All numbers are computed in the browser. No pretrained model is used; where a demonstration is simplified (hand-picked attention weights, illustrative scaling curves), the page says so.

## Using it

Open `index.html` in any modern browser. There is no build step, and every page also works opened straight from disk. The only network request loads the KaTeX maths renderer from a CDN; offline, equations show as LaTeX source and everything else still works.

To serve it the way GitHub Pages does, run `python -m http.server 8000` and open http://localhost:8000.

## Project layout

- `index.html`, `part-a.html` to `part-d.html`, `interview.html`, `reference.html`: the pages.
- `assets/css/` and `assets/js/`: shared styles and scripts; each part page loads only its own widgets.
- `snippets/`: the source of every "In Python" box.
- `tools/`: checks, below.

## Checking changes

- `python tools/snippets.py --check`: every snippet still prints the output shown on the page (needs NumPy and PyTorch). Run it without `--check` to refresh the boxes after editing a snippet.
- `node tools/test-model.js`: the toy model still trains to a validation loss of about 0.31.
- `tools/check.html`: open in Chrome started with `--allow-file-access-from-files`. It loads every page at phone, tablet and desktop widths on both paths and reports overflow, clipped text and unrendered maths. It should end with "done: no problems".

## Deployment

`.github/workflows/pages.yml` publishes the pages and `assets/` on every push to `main` or `master`.

One-time setup: in the repository, go to **Settings → Pages → Build and deployment** and set **Source** to **GitHub Actions**.

## Caveats

- Facts about the 2017 paper (BLEU scores, model sizes, training times) were written from memory; verify them against the paper before relying on them.
- GPU figures and open-weight model facts (sizes, context lengths, licences, Ollama and vLLM behaviour) were checked against vendor pages and model cards in November 2026. New models and versions appear every few months.
- Tested in Chrome on desktop and at phone width. Not yet tested on Safari, Firefox or with a screen reader.
