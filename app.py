"""
app.py  —  Prompt Word Heatmap · Gradio UI
------------------------------------------
Run in Colab / Kaggle:
    import matplotlib; matplotlib.use("Agg")
    !python app.py
"""

import matplotlib
matplotlib.use("Agg")

import gradio as gr
import torch
from heatmap_generator import HeatmapGenerator

# ─── Models ───────────────────────────────────────────────────
MODELS = {
    "Stable Diffusion 1.5"   : "runwayml/stable-diffusion-v1-5",
    "Stable Diffusion 2.1"   : "stabilityai/stable-diffusion-2-1",
    "DreamShaper"            : "Lykon/DreamShaper",
    "Realistic Vision"       : "SG161222/Realistic_Vision_V1.4",
    "OpenJourney"            : "prompthero/openjourney",
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
    overlay  = result.overlay_pil(alpha=float(alpha), top_k=int(top_k))
    texthmap = result.text_heatmap_pil(top_k=int(top_k))
    rows     = result.top_pairs(top_k=int(top_k))
    table    = "\n".join(f"| {w} | {s:.4f} |" for _, w, s in rows)
    md       = f"| Word | Score |\n|---|---|\n{table}"
    return overlay, texthmap, md

# ─── CSS ──────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

:root {
  --bg:        #08080E;
  --surface:   #101018;
  --surface2:  #16161F;
  --border:    #23232F;
  --accent:    #F97316;
  --accent2:   #7C3AED;
  --text:      #E4E4EF;
  --muted:     #6B6B80;
  --success:   #22C55E;
  --radius:    12px;
  --font-head: 'Space Grotesk', sans-serif;
  --font-body: 'Inter', sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}

/* ── Hide Gradio footer ── */
footer { display: none !important; }

/* ── Header ── */
#app-header {
  text-align: center;
  padding: 48px 24px 32px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 32px;
}
#app-header h1 {
  font-family: var(--font-head);
  font-size: 2.4rem;
  font-weight: 700;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #F97316 0%, #FBBF24 50%, #7C3AED 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 10px;
}
#app-header p {
  color: var(--muted);
  font-size: 1rem;
  margin: 0;
  font-family: var(--font-body);
}

/* ── Section cards ── */
.card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 24px;
  margin-bottom: 16px;
}
.card-label {
  font-family: var(--font-head);
  font-size: 0.7rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 14px;
}

/* ── Gradio block overrides ── */
.gr-group, .gr-box, div[class*="block"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* ── Labels ── */
label span, .gr-form label {
  font-family: var(--font-body) !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  letter-spacing: 0.02em !important;
}

/* ── Textbox ── */
textarea, input[type="text"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.9rem !important;
  padding: 12px 14px !important;
  transition: border-color 0.2s !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--accent) !important;
  outline: none !important;
  box-shadow: 0 0 0 3px rgba(249,115,22,0.12) !important;
}

/* ── Sliders ── */
input[type="range"] {
  accent-color: var(--accent) !important;
}
.gr-slider input[type="number"], input[type="number"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  color: var(--text) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.85rem !important;
}

/* ── Dropdown ── */
.gr-dropdown select, select {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
  font-family: var(--font-body) !important;
}

/* ── Buttons ── */
button {
  font-family: var(--font-head) !important;
  font-weight: 600 !important;
  border-radius: 8px !important;
  transition: all 0.2s !important;
  cursor: pointer !important;
}

#load-btn {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  padding: 10px 20px !important;
}
#load-btn:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

#generate-btn {
  background: linear-gradient(135deg, #F97316, #EA580C) !important;
  border: none !important;
  color: white !important;
  font-size: 1rem !important;
  padding: 14px 28px !important;
  width: 100% !important;
  letter-spacing: 0.02em !important;
  box-shadow: 0 4px 24px rgba(249,115,22,0.3) !important;
}
#generate-btn:hover {
  background: linear-gradient(135deg, #FB923C, #F97316) !important;
  box-shadow: 0 6px 32px rgba(249,115,22,0.45) !important;
  transform: translateY(-1px) !important;
}
#generate-btn:active {
  transform: translateY(0) !important;
}

/* ── Status bar ── */
#model-status textarea, #model-status {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--success) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.82rem !important;
  padding: 10px 14px !important;
  min-height: unset !important;
}

/* ── Output images ── */
.gr-image img {
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
}
.gr-image {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

/* ── Score table ── */
#score-table {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px 20px !important;
}
#score-table table {
  width: 100% !important;
  border-collapse: collapse !important;
  font-family: var(--font-mono) !important;
  font-size: 0.85rem !important;
}
#score-table th {
  color: var(--muted) !important;
  font-weight: 500 !important;
  text-align: left !important;
  padding: 6px 12px !important;
  border-bottom: 1px solid var(--border) !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}
#score-table td {
  padding: 6px 12px !important;
  color: var(--text) !important;
  border-bottom: 1px solid var(--border) !important;
}
#score-table tr:last-child td { border-bottom: none !important; }
#score-table tr:hover td { background: var(--surface2) !important; }

/* ── Examples ── */
.gr-samples table {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}
.gr-samples td {
  color: var(--muted) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.82rem !important;
}
.gr-samples tr:hover td { color: var(--text) !important; }

/* ── Divider ── */
.divider {
  height: 1px;
  background: var(--border);
  margin: 8px 0 24px;
}

/* ── Output section labels ── */
.output-label {
  font-family: var(--font-head);
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 10px;
}
"""

# ─── UI ───────────────────────────────────────────────────────
with gr.Blocks(css=CSS, theme=gr.themes.Base(), title="Prompt Word Heatmap") as demo:

    # ── Header ────────────────────────────────────────────────
    gr.HTML("""
    <div id="app-header">
      <h1>Prompt Word Heatmap</h1>
      <p>See exactly which words in your prompt shape which regions of the generated image —<br>
         extracted from Stable Diffusion's cross-attention layers, no extra generation needed.</p>
    </div>
    """)

    with gr.Row(equal_height=False):

        # ── Left column: controls ──────────────────────────────
        with gr.Column(scale=1, min_width=320):

            # Model card
            gr.HTML('<div class="card-label">① Model</div>')
            with gr.Group():
                model_dd = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],
                    label="",
                    container=False,
                )
                load_btn = gr.Button("Load model", elem_id="load-btn")
                status   = gr.Textbox(
                    value="No model loaded — click Load model",
                    label="", show_label=False,
                    interactive=False,
                    elem_id="model-status",
                )

            gr.HTML('<div class="divider"></div><div class="card-label">② Prompt</div>')
            prompt = gr.Textbox(
                label="",
                placeholder="a golden retriever playing in the snow...",
                lines=3,
                container=False,
            )

            gr.HTML('<div class="divider"></div><div class="card-label">③ Parameters</div>')
            with gr.Row():
                seed  = gr.Slider(0, 9999, value=42,  step=1,   label="Seed")
                steps = gr.Slider(10, 50,  value=20,  step=5,   label="Steps")
            with gr.Row():
                guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5,  label="Guidance")
                alpha    = gr.Slider(0.1, 0.9,  value=0.55, step=0.05, label="Overlay opacity")
            top_k = gr.Slider(3, 20, value=10, step=1,
                              label="Top-K words  (reduces panels for long prompts)")

            gr.HTML('<div class="divider"></div>')
            gen_btn = gr.Button("Generate heatmap →", elem_id="generate-btn")

            # Examples
            gr.HTML('<div class="divider"></div><div class="card-label">Quick examples</div>')
            gr.Examples(
                examples=[
                    ["Stable Diffusion 1.5", "a golden retriever playing in the snow",              42,  20, 7.5, 0.55, 8],
                    ["Stable Diffusion 1.5", "a red sports car on a highway at night",              7,   20, 7.5, 0.55, 8],
                    ["Stable Diffusion 1.5", "an astronaut riding a horse on the moon",             123, 20, 7.5, 0.55, 8],
                    ["Stable Diffusion 1.5", "a cozy wooden cabin in an autumn forest at sunset",   0,   20, 8.0, 0.55, 10],
                    ["Stable Diffusion 1.5", "a futuristic neon city with flying cars at dusk",     99,  20, 7.5, 0.55, 10],
                ],
                inputs=[model_dd, prompt, seed, steps, guidance, alpha, top_k],
                label="",
            )

        # ── Right column: outputs ─────────────────────────────
        with gr.Column(scale=2):

            gr.HTML('<div class="card-label">Spatial overlay  <span style="color:#6B6B80;font-weight:400;text-transform:none;letter-spacing:0">— per-word attention overlaid on the image</span></div>')
            overlay_out = gr.Image(label="", show_label=False, container=True)

            gr.HTML('<div class="divider"></div><div class="card-label">Text heatmap  <span style="color:#6B6B80;font-weight:400;text-transform:none;letter-spacing:0">— top-K words ranked by influence</span></div>')
            texthmap_out = gr.Image(label="", show_label=False, container=True)

            gr.HTML('<div class="divider"></div><div class="card-label">Attention scores</div>')
            score_md = gr.Markdown(elem_id="score-table")

    # ── Events ────────────────────────────────────────────────
    load_btn.click(fn=load_model, inputs=[model_dd], outputs=[status])
    gen_btn .click(fn=generate,
                   inputs=[model_dd, prompt, seed, steps, guidance, alpha, top_k],
                   outputs=[overlay_out, texthmap_out, score_md])

if __name__ == "__main__":
    demo.launch(share=True)
