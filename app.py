"""
app.py  —  PromptLens · Gradio UI
------------------------------------------
Run in Colab / Kaggle:
    !python app.py
"""

import matplotlib
matplotlib.use("Agg")

import gradio as gr
import torch
from heatmap_generator import HeatmapGenerator, _pos_map

# ─── Models ───────────────────────────────────────────────────
MODELS = {
    "Stable Diffusion 1.5"              : "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion 2.1"              : "stabilityai/stable-diffusion-2-1",
    "SDXL 1.0  ✦ best quality"         : "stabilityai/stable-diffusion-xl-base-1.0",
    "DreamShaper  ✦ artistic"           : "Lykon/DreamShaper",
    "Realistic Vision  ✦ photorealistic": "SG161222/Realistic_Vision_V1.4",
    "OpenJourney  ✦ Midjourney-style"   : "prompthero/openjourney",
}

_gen             = None
_loaded_model_id = None

def _get_gen(model_id):
    global _gen, _loaded_model_id
    if _gen is None or _loaded_model_id != model_id:
        _gen             = HeatmapGenerator(model_id=model_id, steps=30, capture_n=10)
        _loaded_model_id = model_id
    return _gen

def load_model(model_label):
    model_id = MODELS[model_label]
    try:
        _get_gen(model_id)
        mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        return f"✅  {model_label} loaded  ·  {mem:.1f} GB GPU"
    except Exception as e:
        return f"❌  {e}"

def generate(model_label, prompt, seed, steps, guidance, alpha, top_k):
    if not prompt.strip():
        raise gr.Error("Enter a prompt first.")

    gen    = _get_gen(MODELS[model_label])
    result = gen.run(prompt=prompt, seed=int(seed),
                     steps=int(steps), guidance=float(guidance))

    # ── Word view outputs ──────────────────────────────────
    overlay     = result.overlay_pil(alpha=float(alpha), top_k=int(top_k))
    text_hmap   = result.text_heatmap_pil(top_k=int(top_k))
    pm          = _pos_map(prompt)
    word_rows   = result.top_pairs(top_k=int(top_k))
    word_table  = "| # | Word | POS | Score |\n|---|---|---|---|\n"
    word_table += "\n".join(
        f"| {i+1} | **{w}** | {pm.get(w.lower(), '—')} | `{s:.4f}` |"
        for i, (_, w, s) in enumerate(word_rows)
    )

    # ── POS view outputs ───────────────────────────────────
    pos_overlay = result.pos_overlay_pil(alpha=float(alpha))
    pos_hmap    = result.pos_heatmap_pil()
    pos_data    = result.pos_totals()
    pos_table   = "| POS Category | Words | Total Score | Avg Score |\n|---|---|---|---|\n"
    pos_table  += "\n".join(
        f"| **{cat}** | {', '.join(d['words'])} | `{d['total']:.4f}` | `{d['avg']:.4f}` |"
        for cat, d in pos_data.items()
    )

    return overlay, text_hmap, word_table, pos_overlay, pos_hmap, pos_table


# ─── CSS ──────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

:root {
  --bg:       #08080E;
  --surface:  #101018;
  --surface2: #16161F;
  --border:   #23232F;
  --accent:   #F97316;
  --accent2:  #7C3AED;
  --text:     #E4E4EF;
  --muted:    #6B6B80;
  --success:  #22C55E;
  --radius:   12px;
  --fh: 'Space Grotesk', sans-serif;
  --fb: 'Inter', sans-serif;
  --fm: 'JetBrains Mono', monospace;
}

body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; font-family: var(--fb) !important; }
footer { display: none !important; }

#app-header { text-align: center; padding: 44px 24px 28px; border-bottom: 1px solid var(--border); margin-bottom: 28px; }
#app-header h1 { font-family: var(--fh); font-size: 2.4rem; font-weight: 700; letter-spacing: -0.02em;
  background: linear-gradient(135deg, #F97316 0%, #FBBF24 45%, #7C3AED 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 8px; }
#app-header p { color: var(--muted); font-size: 0.97rem; margin: 0; }

.card-label { font-family: var(--fh); font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 10px; }
.divider { height: 1px; background: var(--border); margin: 14px 0 18px; }

.gr-group, .gr-box, div[class*="block"] { background: transparent !important; border: none !important; box-shadow: none !important; }

label span, .gr-form label { font-family: var(--fb) !important; font-size: 0.82rem !important; font-weight: 500 !important; color: var(--muted) !important; }

textarea, input[type="text"] { background: var(--surface2) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; color: var(--text) !important; font-family: var(--fm) !important;
  font-size: 0.9rem !important; padding: 12px 14px !important; transition: border-color 0.2s !important; }
textarea:focus, input[type="text"]:focus { border-color: var(--accent) !important; outline: none !important;
  box-shadow: 0 0 0 3px rgba(249,115,22,0.12) !important; }

input[type="range"] { accent-color: var(--accent) !important; }
input[type="number"] { background: var(--surface2) !important; border: 1px solid var(--border) !important;
  border-radius: 6px !important; color: var(--text) !important; font-family: var(--fm) !important; font-size: 0.85rem !important; }

select { background: var(--surface2) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; color: var(--text) !important; font-family: var(--fb) !important; }

button { font-family: var(--fh) !important; font-weight: 600 !important; border-radius: 8px !important; transition: all 0.2s !important; cursor: pointer !important; }

#load-btn { background: var(--surface2) !important; border: 1px solid var(--border) !important; color: var(--text) !important; }
#load-btn:hover { border-color: var(--accent) !important; color: var(--accent) !important; }

#generate-btn { background: linear-gradient(135deg, #F97316, #EA580C) !important; border: none !important;
  color: white !important; font-size: 1rem !important; padding: 14px 28px !important; width: 100% !important;
  box-shadow: 0 4px 24px rgba(249,115,22,0.28) !important; }
#generate-btn:hover { background: linear-gradient(135deg, #FB923C, #F97316) !important;
  box-shadow: 0 6px 32px rgba(249,115,22,0.42) !important; transform: translateY(-1px) !important; }
#generate-btn:active { transform: translateY(0) !important; }

#model-status textarea { background: var(--surface2) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; color: var(--success) !important; font-family: var(--fm) !important;
  font-size: 0.82rem !important; padding: 10px 14px !important; min-height: unset !important; }

/* Tabs */
.tab-nav { border-bottom: 1px solid var(--border) !important; background: transparent !important; }
.tab-nav button { background: transparent !important; border: none !important; color: var(--muted) !important;
  font-family: var(--fh) !important; font-size: 0.85rem !important; padding: 10px 20px !important;
  border-bottom: 2px solid transparent !important; border-radius: 0 !important; }
.tab-nav button.selected { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }
.tab-nav button:hover { color: var(--text) !important; }

/* Score tables */
#word-score-table, #pos-score-table { background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important; padding: 14px 18px !important; }
#word-score-table table, #pos-score-table table { width: 100% !important; border-collapse: collapse !important;
  font-family: var(--fm) !important; font-size: 0.83rem !important; }
#word-score-table th, #pos-score-table th { color: var(--muted) !important; font-weight: 500 !important;
  text-align: left !important; padding: 5px 10px !important; border-bottom: 1px solid var(--border) !important;
  font-size: 0.72rem !important; letter-spacing: 0.06em !important; text-transform: uppercase !important; }
#word-score-table td, #pos-score-table td { padding: 5px 10px !important; color: var(--text) !important; border-bottom: 1px solid var(--border) !important; }
#word-score-table tr:last-child td, #pos-score-table tr:last-child td { border-bottom: none !important; }
#word-score-table tr:hover td, #pos-score-table tr:hover td { background: var(--surface2) !important; }

.gr-image { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
"""

# ─── UI ───────────────────────────────────────────────────────
with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="PromptLens") as demo:

    gr.HTML("""
    <div id="app-header">
      <h1>PromptLens</h1>
      <p>See which words in your prompt shape which regions of the generated image.<br>
         Switch between <b>Word view</b> and <b>Part-of-Speech view</b> to explore attention at different levels.</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left: controls ─────────────────────────────────────
        with gr.Column(scale=1, min_width=300):

            gr.HTML('<div class="card-label">① Model</div>')
            model_dd = gr.Dropdown(choices=list(MODELS.keys()),
                                   value=list(MODELS.keys())[0],
                                   label="", container=False)
            load_btn = gr.Button("Load model", elem_id="load-btn")
            status   = gr.Textbox(value="No model loaded — click Load model",
                                  label="", show_label=False,
                                  interactive=False, elem_id="model-status")

            gr.HTML('<div class="divider"></div><div class="card-label">② Prompt</div>')
            prompt = gr.Textbox(label="", lines=3, container=False,
                                placeholder="a golden retriever playing in the snow…")

            gr.HTML('<div class="divider"></div><div class="card-label">③ Parameters</div>')
            with gr.Row():
                seed  = gr.Slider(0, 9999, value=42,  step=1,    label="Seed")
                steps = gr.Slider(10, 50,  value=20,  step=5,    label="Steps")
            with gr.Row():
                guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5,  label="Guidance")
                alpha    = gr.Slider(0.1, 0.9,  value=0.55, step=0.05, label="Overlay opacity")
            top_k = gr.Slider(3, 20, value=10, step=1,
                              label="Top-K words  (word view only)")

            gr.HTML('<div class="divider"></div>')
            gen_btn = gr.Button("Generate  →", elem_id="generate-btn")

            gr.HTML('<div class="divider"></div><div class="card-label">Quick examples</div>')
            gr.Examples(
                examples=[
                    [list(MODELS.keys())[0], "a golden retriever playing in the snow",               42,  20, 7.5, 0.55, 8],
                    [list(MODELS.keys())[0], "a red sports car on a highway at night",               7,   20, 7.5, 0.55, 8],
                    [list(MODELS.keys())[0], "an astronaut riding a horse on the moon",              123, 20, 7.5, 0.55, 8],
                    [list(MODELS.keys())[0], "a cozy wooden cabin in an autumn forest at sunset",    0,   20, 8.0, 0.55, 10],
                    [list(MODELS.keys())[0], "a futuristic neon city with flying cars at dusk",      99,  20, 7.5, 0.55, 10],
                ],
                inputs=[model_dd, prompt, seed, steps, guidance, alpha, top_k],
                label="",
            )

        # ── Right: tabbed outputs ──────────────────────────────
        with gr.Column(scale=2):

            with gr.Tabs():

                # ── Tab 1: Word View ───────────────────────────
                with gr.Tab("🔤  Word View"):
                    gr.HTML('<div class="card-label" style="margin-top:12px">Spatial overlay  <span style="color:#6B6B80;font-weight:400;text-transform:none;letter-spacing:0">— top-K words overlaid on image</span></div>')
                    overlay_out   = gr.Image(label="", show_label=False)

                    gr.HTML('<div class="divider"></div><div class="card-label">Text heatmap  <span style="color:#6B6B80;font-weight:400;text-transform:none;letter-spacing:0">— blue low · red high attention</span></div>')
                    text_hmap_out = gr.Image(label="", show_label=False)

                    gr.HTML('<div class="divider"></div><div class="card-label">Word scores</div>')
                    word_score_md = gr.Markdown(elem_id="word-score-table")

                # ── Tab 2: POS View ────────────────────────────
                with gr.Tab("🏷️  Part-of-Speech View"):
                    gr.HTML("""
                    <div style="background:#16161F;border:1px solid #23232F;border-radius:10px;
                                padding:12px 16px;margin:12px 0 16px;font-size:0.82rem;color:#6B6B80;line-height:1.6">
                      Words grouped by grammatical role.
                      <b style="color:#F97316">Nouns</b> · <b style="color:#7C3AED">Verbs</b> ·
                      <b style="color:#06B6D4">Adjectives</b> · <b style="color:#22C55E">Adverbs</b> ·
                      <b style="color:#6B7280">Function words</b><br>
                      The overlay shows the combined heatmap of all words in each category.
                    </div>
                    """)

                    gr.HTML('<div class="card-label">POS spatial overlay  <span style="color:#6B6B80;font-weight:400;text-transform:none;letter-spacing:0">— one panel per grammatical category</span></div>')
                    pos_overlay_out = gr.Image(label="", show_label=False)

                    gr.HTML('<div class="divider"></div><div class="card-label">POS breakdown  <span style="color:#6B6B80;font-weight:400;text-transform:none;letter-spacing:0">— words colored by role + total attention bars</span></div>')
                    pos_hmap_out    = gr.Image(label="", show_label=False)

                    gr.HTML('<div class="divider"></div><div class="card-label">POS scores</div>')
                    pos_score_md    = gr.Markdown(elem_id="pos-score-table")

    # ── Wire events ───────────────────────────────────────────
    load_btn.click(fn=load_model, inputs=[model_dd], outputs=[status])

    gen_btn.click(
        fn      = generate,
        inputs  = [model_dd, prompt, seed, steps, guidance, alpha, top_k],
        outputs = [overlay_out, text_hmap_out, word_score_md,
                   pos_overlay_out, pos_hmap_out, pos_score_md],
    )

if __name__ == "__main__":
    demo.launch(share=True)
