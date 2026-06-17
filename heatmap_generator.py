"""
heatmap_generator.py
--------------------
Load once, run as many prompts as you want.

Usage:
    from heatmap_generator import HeatmapGenerator

    gen = HeatmapGenerator()
    result = gen.run("a cat in the snow")
    result.show()                        # top 10 words by default
    result.show(top_k=5)                 # only top 5
    result.save("out/")
"""

import gc
import io
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")                    # headless — prevents Colab/Kaggle hangs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import Attention


# ─────────────────────────────────────────────────────────────
#  Available models (add more here anytime)
# ─────────────────────────────────────────────────────────────

AVAILABLE_MODELS = {
    "SD 1.5  (fast, 4GB)"        : "runwayml/stable-diffusion-v1-5",
    "SD 2.1  (better quality)"   : "stabilityai/stable-diffusion-2-1",
    "DreamShaper (artistic)"     : "Lykon/DreamShaper",
    "Realistic Vision (photo)"   : "SG161222/Realistic_Vision_V1.4",
}


# ─────────────────────────────────────────────────────────────
#  Internal: custom attention processor
# ─────────────────────────────────────────────────────────────

class _CapturingProcessor:
    def __init__(self, store, layer_name):
        self.store      = store
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        is_cross = encoder_hidden_states is not None
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        kv    = encoder_hidden_states if is_cross else hidden_states
        key   = attn.to_k(kv)
        value = attn.to_v(kv)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key   = key  .view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        scale        = head_dim ** -0.5
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)

        if is_cross and self.store._should_capture():
            self.store._maps[self.layer_name].append(
                attn_weights.detach().cpu().half()
            )

        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        ).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


# ─────────────────────────────────────────────────────────────
#  Internal: attention store
# ─────────────────────────────────────────────────────────────

class _AttentionStore:
    def __init__(self, capture_last_n, total_steps):
        self._maps     = defaultdict(list)
        self._original = {}
        self._step     = 0
        self._start_at = max(0, total_steps - capture_last_n)

    def _should_capture(self):
        return self._step >= self._start_at

    def on_step_end(self, pipe, step_index, timestep, callback_kwargs):
        self._step += 1
        return callback_kwargs

    def register(self, unet):
        new_procs = {}
        for name, proc in unet.attn_processors.items():
            if "attn2" in name:
                new_procs[name] = _CapturingProcessor(self, name)
            else:
                new_procs[name] = proc
        self._original = unet.attn_processors.copy()
        unet.set_attn_processor(new_procs)

    def restore(self, unet):
        unet.set_attn_processor(self._original)

    def aggregate(self, token_count, target_res=16, out_size=64):
        all_maps = []
        for layer_maps in self._maps.values():
            for attn in layer_maps:
                b       = attn.shape[0]
                cond    = attn[b // 2:].mean(dim=1)
                spatial = cond.shape[1]
                side    = int(spatial ** 0.5)
                if side * side == spatial:
                    all_maps.append((side, cond[0]))

        if not all_maps:
            raise RuntimeError("No attention maps captured.")

        best     = min(set(s for s, _ in all_maps), key=lambda r: abs(r - target_res))
        selected = torch.stack([m for s, m in all_maps if s == best]).float()
        avg      = selected.mean(0)[:, :token_count]
        h = w    = best
        avg      = avg.reshape(h, w, -1).permute(2, 0, 1).unsqueeze(0)
        avg      = F.interpolate(avg, size=(out_size, out_size),
                                 mode="bilinear", align_corners=False).squeeze(0)
        mn, mx   = avg.amin(dim=(-2,-1), keepdim=True), avg.amax(dim=(-2,-1), keepdim=True)
        return ((avg - mn) / (mx - mn + 1e-8)).numpy()


# ─────────────────────────────────────────────────────────────
#  Result object
# ─────────────────────────────────────────────────────────────

_SPECIAL = {"<|startoftext|>", "<|endoftext|>", "<pad>"}

@dataclass
class HeatmapResult:
    image:    Image.Image
    tokens:   list
    heatmaps: np.ndarray   # (N_tokens, 64, 64)
    prompt:   str

    # ── derived helpers ──────────────────────────────────────

    @property
    def words(self):
        return [t.replace("Ġ", " ").replace("</w>", "").strip()
                for t in self.tokens if t not in _SPECIAL]

    @property
    def word_indices(self):
        return [i for i, t in enumerate(self.tokens) if t not in _SPECIAL]

    @property
    def scores(self):
        return [float(self.heatmaps[i].mean()) for i in self.word_indices]

    def top_pairs(self, top_k=10):
        """
        Return (word_index, word, score) tuples sorted by score descending,
        limited to top_k. If top_k is None, returns all words.
        """
        triplets = list(zip(self.word_indices, self.words, self.scores))
        triplets.sort(key=lambda x: x[2], reverse=True)
        if top_k is not None:
            triplets = triplets[:top_k]
        # re-sort by original position so layout reads left→right
        triplets.sort(key=lambda x: x[0])
        return triplets

    # ── plots ────────────────────────────────────────────────

    def plot_overlay(self, alpha=0.55, top_k=10, ncols=4, figsize=None) -> plt.Figure:
        """
        Grid of per-word attention overlays.
        top_k : show only the top-K most important words (None = all words)
        """
        pairs   = [(i, w) for i, w, s in self.top_pairs(top_k)]
        n       = len(pairs)
        ncols   = min(ncols, n + 1)
        nrows   = 1 + (n + ncols - 2) // (ncols - 1) if ncols > 1 else n + 1
        figsize = figsize or (4 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes      = np.array(axes).flatten()
        img512    = np.array(self.image.resize((512, 512)))

        axes[0].imshow(img512)
        axes[0].set_title("Generated image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        for panel, (tok_i, word) in enumerate(pairs, start=1):
            ax      = axes[panel]
            heat    = Image.fromarray((self.heatmaps[tok_i] * 255).astype(np.uint8)) \
                           .resize((512, 512), Image.BILINEAR)
            colored = cm.inferno(np.array(heat) / 255.0)[..., :3]
            blended = np.clip((1 - alpha) * img512 / 255.0 + alpha * colored, 0, 1)
            ax.imshow(blended)
            ax.set_title(f'"{word}"', fontsize=10)
            ax.axis("off")

        for ax in axes[panel + 1:]:
            ax.axis("off")

        top_label = f"top {top_k}" if top_k else "all"
        fig.suptitle(f'Cross-attention overlay ({top_label} words) — "{self.prompt}"',
                     fontsize=10, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_text_heatmap(self, top_k=10, figsize=(12, 4)) -> plt.Figure:
        """
        Colored word-token bar showing ONLY the top-K words by attention score,
        sorted left→right by original position in the prompt.
        Blue = low attention, Red = high attention.
        """
        triplets = self.top_pairs(top_k)   # already sorted by prompt position

        max_s = max(s for _, _, s in triplets) or 1.0
        norm  = Normalize(vmin=0, vmax=max_s)
        cmap  = plt.get_cmap("RdBu_r")

        fig, axes = plt.subplots(1, 2, figsize=figsize,
                                 gridspec_kw={"width_ratios": [1, 1.6]})
        axes[0].imshow(self.image)
        axes[0].axis("off")
        axes[0].set_title("Generated image", fontsize=10)

        ax = axes[1]
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        top_label = f"Top {len(triplets)} words" if top_k else "All words"
        ax.set_title(f"Text Heatmap  —  {top_label} by attention score",
                     fontsize=10, loc="left")

        n     = len(triplets)
        box_w = 1.0 / n

        for i, (_, word, score) in enumerate(triplets):
            x_c   = (i + 0.5) * box_w
            color = cmap(norm(score))

            rect = mpatches.FancyBboxPatch(
                (i * box_w + 0.008, 0.38), box_w - 0.016, 0.42,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="gray", linewidth=0.6,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)

            bright     = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            text_color = "black" if bright > 0.5 else "white"
            ax.text(x_c, 0.61, word, ha="center", va="center",
                    fontsize=10, fontweight="500",
                    color=text_color, transform=ax.transAxes)
            ax.text(x_c, 0.24, f"{score:.3f}", ha="center", va="center",
                    fontsize=8, color="#444", transform=ax.transAxes)

        fig.tight_layout()
        return fig

    # ── convenience ─────────────────────────────────────────

    def show(self, top_k=10):
        self.plot_overlay(top_k=top_k).show()
        self.plot_text_heatmap(top_k=top_k).show()

    def save(self, folder=".", prefix=None, top_k=10):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        tag = prefix or self.prompt[:30].replace(" ", "_")
        p1  = folder / f"{tag}_overlay.png"
        p2  = folder / f"{tag}_text_heatmap.png"
        self.plot_overlay(top_k=top_k)     .savefig(p1, dpi=130, bbox_inches="tight")
        self.plot_text_heatmap(top_k=top_k).savefig(p2, dpi=130, bbox_inches="tight")
        plt.close("all")
        print(f"Saved:\n  {p1}\n  {p2}")
        return str(p1), str(p2)

    def overlay_pil(self, alpha=0.55, top_k=10) -> Image.Image:
        fig = self.plot_overlay(alpha=alpha, top_k=top_k)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def text_heatmap_pil(self, top_k=10) -> Image.Image:
        fig = self.plot_text_heatmap(top_k=top_k)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()


# ─────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────

class HeatmapGenerator:
    """
    Load a Stable Diffusion pipeline once, then call .run() with
    as many prompts as you like. Call .switch_model() to swap models
    without restarting — old model is unloaded from VRAM first.

    Example
    -------
    gen = HeatmapGenerator()
    r1  = gen.run("a cat in the snow")
    r1.show(top_k=10)

    gen.switch_model("stabilityai/stable-diffusion-2-1")
    r2  = gen.run("a futuristic city at dusk")
    r2.show()
    """

    def __init__(
        self,
        model_id:  str = "runwayml/stable-diffusion-v1-5",
        device:    str = None,
        steps:     int = 30,
        capture_n: int = 10,
    ):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.steps     = steps
        self.capture_n = capture_n
        self.model_id  = None
        self.pipe      = None
        self._load(model_id)

    # ── private ──────────────────────────────────────────────

    def _load(self, model_id: str):
        # Unload existing model from VRAM before loading new one
        if self.pipe is not None:
            print(f"Unloading {self.model_id}…")
            del self.pipe
            self.pipe = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        print(f"Loading {model_id} on {self.device}…")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(self.device)
        self.pipe.enable_attention_slicing()
        self.pipe.enable_vae_slicing()
        self.model_id = model_id
        print(f"Ready — {model_id}")

    # ── public API ───────────────────────────────────────────

    def switch_model(self, model_id: str):
        """
        Swap to a different SD model. Unloads the current one from VRAM first.
        model_id can be a HuggingFace repo id or a key from AVAILABLE_MODELS.
        """
        # Accept friendly name or raw model id
        resolved = AVAILABLE_MODELS.get(model_id, model_id)
        if resolved == self.model_id:
            print(f"Already loaded: {self.model_id}")
            return
        self._load(resolved)

    @property
    def current_model(self) -> str:
        """Friendly name of the currently loaded model."""
        for name, mid in AVAILABLE_MODELS.items():
            if mid == self.model_id:
                return name
        return self.model_id

    def run(
        self,
        prompt:    str,
        seed:      int   = 42,
        steps:     int   = None,
        guidance:  float = 7.5,
        capture_n: int   = None,
    ) -> HeatmapResult:
        steps     = steps     or self.steps
        capture_n = capture_n or self.capture_n

        tokenizer  = self.pipe.tokenizer
        tokens     = tokenizer.encode(prompt)
        token_strs = tokenizer.convert_ids_to_tokens(tokens)

        store = _AttentionStore(capture_last_n=capture_n, total_steps=steps)
        store.register(self.pipe.unet)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        result    = self.pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            callback_on_step_end=store.on_step_end,
        )
        image    = result.images[0]
        heatmaps = store.aggregate(len(tokens))
        store.restore(self.pipe.unet)

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return HeatmapResult(
            image=image,
            tokens=token_strs,
            heatmaps=heatmaps,
            prompt=prompt,
        )
