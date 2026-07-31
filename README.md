<div align="center">

<br/>

```
██████╗ ██████╗  ██████╗ ███╗   ███╗██████╗ ████████╗██╗     ███████╗███╗   ██╗███████╗
██╔══██╗██╔══██╗██╔═══██╗████╗ ████║██╔══██╗╚══██╔══╝██║     ██╔════╝████╗  ██║██╔════╝
██████╔╝██████╔╝██║   ██║██╔████╔██║██████╔╝   ██║   ██║     █████╗  ██╔██╗ ██║███████╗
██╔═══╝ ██╔══██╗██║   ██║██║╚██╔╝██║██╔═══╝    ██║   ██║     ██╔══╝  ██║╚██╗██║╚════██║
██║     ██║  ██║╚██████╔╝██║ ╚═╝ ██║██║        ██║   ███████╗███████╗██║ ╚████║███████║
╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝        ╚═╝   ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝
```

### *See exactly which words shape which pixels.*

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/🤗_Diffusers-0.27+-FFD21F?style=for-the-badge)](https://huggingface.co/docs/diffusers)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> PromptLens hooks into the **cross-attention layers** of Stable Diffusion and maps every word in your prompt to the exact pixels it influenced — extracted live during inference, no extra passes needed.

<br/>

</div>

---

## 📸 Demo

<!-- Replace the blocks below with real screenshots once you've run the app -->

| Spatial Overlay | Text Heatmap | POS Breakdown |
|:---:|:---:|:---:|
| ![Spatial overlay — per-word attention heatmaps overlaid on the generated image](assets/demo_overlay.png) | ![Text heatmap — top-K words color-coded by attention score](assets/demo_textheatmap.png) | ![POS breakdown — words grouped by grammatical role with total attention per category](assets/demo_pos.png) |
| *Per-word attention overlaid on the generated image* | *Top-K words ranked by influence, blue → red* | *Which parts of speech (nouns, verbs, adjectives…) the model leaned on most* |

<br/>

---

## ✨ What Does It Do?

When you write `"a golden retriever playing in the snow"`, PromptLens answers:

- Which pixels did **"retriever"** activate?
- Did **"golden"** influence the fur color or something else?
- How much weight did the model give **"snow"** vs **"playing"**?
- Does the model lean on **nouns** more than **adjectives**? Do **verbs** even matter?

It does this by registering a custom processor on every cross-attention (`attn2`) block in the UNet, collecting the raw attention weight tensors across the final N denoising steps, averaging them, and projecting them back to image space — one 64×64 heatmap per token.

On top of the per-word view, PromptLens also runs a **part-of-speech (POS) analysis**: every word in the prompt is tagged (noun, verb, adjective, …) with spaCy, its attention score is rolled up into that grammatical category, and the result is rendered as its own overlay + bar chart — so instead of "which word mattered," you can ask "which *kind* of word mattered."

---

## 🗂 Project Structure

```
PromptLens/
│
├── app.py                  # Gradio web UI
├── heatmap_generator.py    # Core library — use this standalone too
│
├── requirements.txt        # Python dependencies
├── LICENSE                 # MIT
├── .gitignore
│
└── assets/                 # Screenshots for this README
    ├── demo_overlay.png
    ├── demo_textheatmap.png
    └── demo_pos.png
```

---

## ⚡ Quickstart

### 1 — Clone

```bash
git clone https://github.com/KnoxCodes/PromptLens.git
cd PromptLens
```

### 2 — Install

```bash
pip install -r requirements.txt
```

> **GPU strongly recommended.** A CUDA GPU with ≥ 6 GB VRAM gives you ~20 s per image.
> CPU works but is very slow (~5–10 min).

### 3 — Run the UI

```bash
python app.py
```

Open the local Gradio URL printed in the terminal. Use `--share` or set `share=True` in `app.py` to get a public link (useful for Colab / Kaggle).

---

## 🐍 Python API

You can also use `HeatmapGenerator` as a standalone library, no UI needed:

```python
from heatmap_generator import HeatmapGenerator

# Load once
gen = HeatmapGenerator(model_id="runwayml/stable-diffusion-v1-5")

# Run on any prompt
result = gen.run("a futuristic city at dusk", seed=42)

# Visualise
result.show()                  # opens overlay + text heatmap
result.show(top_k=5)           # only the 5 most influential words

# Part-of-speech analysis
result.show_pos()              # opens POS overlay + POS breakdown chart
result.pos_totals()            # {'Nouns': {'total': 1.82, 'avg': 0.61, 'words': [...]}, ...}

# Save to disk
result.save("outputs/")        # saves overlay.png, text_heatmap.png, pos_overlay.png, pos_heatmap.png

# Switch models — old model is unloaded from VRAM first
gen.switch_model("stabilityai/stable-diffusion-2-1")
result2 = gen.run("a red sports car on a highway at night", seed=7)
```

### `HeatmapResult` reference

| Attribute / Method | Type | Description |
|---|---|---|
| `.image` | `PIL.Image` | The generated image |
| `.words` | `list[str]` | Clean word strings (special tokens stripped) |
| `.scores` | `list[float]` | Per-word mean attention score |
| `.top_pairs(k)` | `list[tuple]` | `(index, word, score)` — top-k, sorted by prompt position |
| `.overlay_pil(alpha, top_k)` | `PIL.Image` | Spatial heatmap grid as PIL image |
| `.text_heatmap_pil(top_k)` | `PIL.Image` | Color-coded token bar as PIL image |
| `.pos_totals()` | `dict` | `{category: {'total', 'avg', 'words'}}`, sorted by total score |
| `.plot_pos_overlay(alpha)` | `Figure` | One spatial overlay panel per POS category (matplotlib) |
| `.plot_pos_heatmap()` | `Figure` | Words colored by POS + bar chart of totals (matplotlib) |
| `.pos_overlay_pil(alpha)` | `PIL.Image` | POS spatial overlay as PIL image |
| `.pos_heatmap_pil()` | `PIL.Image` | POS breakdown chart as PIL image |
| `.show_pos()` | — | Opens the POS overlay + POS breakdown chart |
| `.save(folder, prefix, top_k)` | `dict[str, Path]` | Saves all four figures (overlay, text heatmap, POS overlay, POS heatmap) |

---

## 🏷️ Part-of-Speech Analysis

Every word in the prompt is tagged with [spaCy](https://spacy.io/) (`en_core_web_sm`) and rolled up into one of seven broad, readable categories — so you can see whether the model is paying attention to *subjects*, *actions*, *descriptors*, or just grammatical glue.

| Category | Includes (spaCy tags) | Color |
|---|---|---|
| **Nouns** | `NOUN`, `PROPN` | 🟧 orange |
| **Verbs** | `VERB`, `AUX` | 🟪 purple |
| **Adjectives** | `ADJ` | 🟦 cyan |
| **Adverbs** | `ADV` | 🟩 green |
| **Function** | `DET`, `ADP`, `CCONJ`, `SCONJ`, `CONJ`, `PART`, `PRON` | ⬜ gray |
| **Numbers** | `NUM` | 🟨 yellow |
| **Other** | `PUNCT`, `SYM`, `X`, `INTJ` | ◻️ slate |

**How it works:**

1. The raw prompt is tagged with spaCy's `en_core_web_sm` model to get one fine-grained POS tag per word.
2. Fine-grained tags are collapsed into the seven broad categories above (e.g. `NOUN` + `PROPN` → **Nouns**).
3. Each CLIP token's attention score (already computed for the word-level view) is matched back to its source word — including partial matching for subword tokens (e.g. the token `"gold"` still matches the tagged word `"golden"`).
4. Scores are summed and averaged per category, and word-level heatmaps within a category are averaged together for the spatial overlay.

**What you get, in both the Python API and the Gradio UI (under the "🏷️ Part-of-Speech View" tab):**

- **POS spatial overlay** — one attention-overlay panel per category (e.g. all "Noun" heatmaps averaged into a single panel), rendered in that category's color.
- **POS breakdown chart** — every word in the prompt colored by its category (opacity ∝ its attention score), plus a bar chart of total attention per category.
- **POS score table** — category, member words, total score, and average score.

The spaCy model downloads automatically (one-time, ~13 MB) the first time POS analysis runs. If spaCy or the model isn't available, POS features degrade gracefully — word-level heatmaps and the Word View tab are unaffected.

---

## 🤖 Supported Models

| Name | HuggingFace ID | Best For |
|---|---|---|
| Stable Diffusion 1.5 | `runwayml/stable-diffusion-v1-5` | Fast baseline, ~4 GB VRAM |
| Stable Diffusion 2.1 | `stabilityai/stable-diffusion-2-1` | Better overall quality |
| SDXL 1.0 | `stabilityai/stable-diffusion-xl-base-1.0` | Best quality (uses CPU offload) |
| DreamShaper | `Lykon/DreamShaper` | Artistic / illustrated style |
| Realistic Vision | `SG161222/Realistic_Vision_V1.4` | Photorealistic output |
| OpenJourney | `prompthero/openjourney` | MidJourney-like aesthetic |

Any SD 1.x / 2.x model that uses the standard `StableDiffusionPipeline` should work — just pass the HuggingFace repo ID to `HeatmapGenerator(model_id=...)`.

---

## 🔬 How It Works

```
                         ┌─────────────────────────────────┐
  "a cat in the snow"    │         CLIP Tokenizer          │
         │               └────────────────┬────────────────┘
         │                                │  token ids
         │               ┌────────────────▼────────────────┐
         │               │        CLIP Text Encoder        │
         │               └────────────────┬────────────────┘
         │                                │  text embeddings
         │               ┌────────────────▼────────────────┐
         │               │    UNet (denoising, T steps)    │
         │               │                                 │
         │               │  ┌──────────────────────────┐   │
         │               │  │  attn2 (cross-attention) │◄──┘
         │               │  │  — captured last N steps │
         │               │  └──────────────┬───────────┘
         │               └─────────────────┼───────────────┘
         │                                 │  [B, heads, spatial, tokens]
         │               ┌─────────────────▼───────────────┐
         │               │  Average heads + steps          │
         │               │  Upsample to 64×64              │
         │               │  Normalize per token            │
         └──────────────►│  Filter special tokens          │
                         └─────────────────┬───────────────┘
                                           │
                         ┌─────────────────▼───────────────┐
                         │   Per-word 64×64 heatmap array  │
                         │   → overlay on generated image  │
                         │   → color-coded text strip      │
                         └─────────────────────────────────┘
```

**Key implementation details:**

- A `_CapturingProcessor` replaces every `attn2` processor in the UNet via `set_attn_processor()`.
- Attention weights are captured only during the **last `capture_n` steps** (default 10) — early steps are too noisy to be meaningful.
- After generation, the original processors are **restored** so the pipeline stays clean for the next call.
- Weights from all captured layers and steps are averaged, then bilinearly upsampled from the native attention resolution (16×16) to 64×64 for display.
- The **POS layer** sits on top of this: it tags the raw prompt with spaCy, maps each CLIP token back to its tagged word, and groups the same per-token heatmaps/scores by grammatical category — no extra model passes required.

---

## 🖥 Running on Colab / Kaggle (Free GPU)

Since Hugging Face Spaces no longer offers free GPU, the easiest way to share a live demo is via Colab or Kaggle with a public share link:

```python
# At the top of a Colab cell
import matplotlib; matplotlib.use("Agg")
!pip install -q diffusers transformers accelerate gradio spacy
!python -m spacy download -q en_core_web_sm

# Then at the bottom of app.py, change:
demo.launch(share=True)   # <-- share=True gives you a public URL
```

The public link stays alive for ~12 hours per session.

---


## 📄 License

MIT — see [LICENSE](LICENSE). Use it, fork it, build on it.

---

<div align="center">

Built with 🔥 by [Knox](https://github.com/KnoxCodes)

*If PromptLens helped you understand your prompts better, consider giving it a ⭐*

</div>
