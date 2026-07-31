import gc
import io
import sys
import subprocess
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from diffusers.models.attention_processor import Attention


# Fine-grained spacy tag → broad readable category
_BROAD_POS = {
    'NOUN':  'Nouns',     'PROPN': 'Nouns',
    'VERB':  'Verbs',     'AUX':   'Verbs',
    'ADJ':   'Adjectives',
    'ADV':   'Adverbs',
    'DET':   'Function',  'ADP':   'Function',
    'CCONJ': 'Function',  'SCONJ': 'Function',
    'CONJ':  'Function',  'PART':  'Function',
    'PRON':  'Function',
    'NUM':   'Numbers',
    'PUNCT': 'Other',     'SYM':   'Other',
    'X':     'Other',     'INTJ':  'Other',
    'OTHER': 'Other',
}

# Color per broad category
_POS_COLORS = {
    'Nouns':       '#F97316',   # orange  — main subjects
    'Verbs':       '#7C3AED',   # purple  — actions
    'Adjectives':  '#06B6D4',   # cyan    — descriptors
    'Adverbs':     '#22C55E',   # green   — modifiers
    'Function':    '#6B7280',   # gray    — grammar glue
    'Numbers':     '#EAB308',   # yellow
    'Other':       '#94A3B8',   # slate
}

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spacy model (one-time)…")
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True, capture_output=True,
            )
            _nlp = spacy.load("en_core_web_sm")
        return _nlp
    except ImportError:
        return None


def _pos_map(prompt: str) -> dict[str, str]:
    """Return {word_lower: broad_category} for every token in prompt."""
    nlp = _get_nlp()
    if nlp is None:
        return {}
    doc = nlp(prompt)
    out = {}
    for token in doc:
        fine  = token.pos_
        broad = _BROAD_POS.get(fine, 'Other')
        out[token.text.lower()] = broad
    return out


def _match_broad(word: str, pm: dict) -> str:
    """Match a (possibly subword) CLIP token to a broad POS category."""
    w = word.lower().strip(".,!?;:'\"")
    if w in pm:
        return pm[w]
    # partial match for subword tokens  e.g. 'gold' → 'golden'
    for key, val in pm.items():
        if w in key or key in w:
            return val
    return 'Other'

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
        hidden_states = (hidden_states
                         .transpose(1, 2)
                         .reshape(batch_size, -1, attn.heads * head_dim)
                         .to(query.dtype))
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states



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
            new_procs[name] = (_CapturingProcessor(self, name)
                               if "attn2" in name else proc)
        self._original = unet.attn_processors.copy()
        unet.set_attn_processor(new_procs)

    def restore(self, unet):
        unet.set_attn_processor(self._original)

    def aggregate(self, token_count, target_res=16, out_size=64):
        all_maps = []
        for layer_maps in self._maps.values():
            for attn in layer_maps:
                b       = attn.shape[0]
                cond    = attn[b // 2:].mean(dim=1)   # (1, spatial, tokens)
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
        return ((avg - mn) / (mx - mn + 1e-8)).numpy()   # (tokens, 64, 64)



_SPECIAL = {"<|startoftext|>", "<|endoftext|>", "<pad>"}


@dataclass
class HeatmapResult:
    image:    Image.Image
    tokens:   list
    heatmaps: np.ndarray    # (N_tokens, 64, 64)
    prompt:   str

    # ── word-level helpers ────────────────────────────────────

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
        """(tok_idx, word, score) sorted by score; re-ordered left→right."""
        trips = sorted(zip(self.word_indices, self.words, self.scores),
                       key=lambda x: -x[2])
        if top_k:
            trips = trips[:top_k]
        trips.sort(key=lambda x: x[0])
        return trips

    # ── POS helpers ───────────────────────────────────────────

    def _word_pos_list(self):
        """[(word, score, broad_pos)] in prompt order."""
        pm = _pos_map(self.prompt)
        return [(w, s, _match_broad(w, pm))
                for w, s in zip(self.words, self.scores)]

    def pos_totals(self) -> dict:
        """
        Returns {broad_pos: {'total': float, 'avg': float, 'words': [str]}}
        sorted by total score descending.
        """
        groups = defaultdict(lambda: {'total': 0.0, 'count': 0, 'words': []})
        for word, score, broad in self._word_pos_list():
            groups[broad]['total']  += score
            groups[broad]['count']  += 1
            groups[broad]['words'].append(word)
        result = {}
        for broad, data in sorted(groups.items(), key=lambda x: -x[1]['total']):
            result[broad] = {
                'total': round(data['total'], 4),
                'avg':   round(data['total'] / max(data['count'], 1), 4),
                'words': data['words'],
            }
        return result

    # ── word-view plots ───────────────────────────────────────

    def plot_overlay(self, alpha=0.55, top_k=10, ncols=4, figsize=None):
        """Grid: generated image + per-word attention overlay (top-K words)."""
        pairs   = [(i, w) for i, w, _ in self.top_pairs(top_k)]
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

        fig.suptitle(f'Word attention — "{self.prompt}"', fontsize=10, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_text_heatmap(self, top_k=10, figsize=(12, 4)):
        """Colored word-box bar (top-K only). Blue=low, Red=high."""
        triplets = self.top_pairs(top_k)
        max_s    = max(s for _, _, s in triplets) or 1.0
        norm     = Normalize(vmin=0, vmax=max_s)
        cmap     = plt.get_cmap("RdBu_r")

        fig, (ax_img, ax_bar) = plt.subplots(
            1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.6]}
        )
        ax_img.imshow(self.image); ax_img.axis("off")
        ax_img.set_title("Generated image", fontsize=10)

        ax_bar.set_xlim(0, 1); ax_bar.set_ylim(0, 1); ax_bar.axis("off")
        ax_bar.set_title(f"Top {len(triplets)} words by attention score",
                         fontsize=10, loc="left")

        n     = len(triplets)
        box_w = 1.0 / n
        for i, (_, word, score) in enumerate(triplets):
            x_c   = (i + 0.5) * box_w
            color = cmap(norm(score))
            rect  = mpatches.FancyBboxPatch(
                (i * box_w + 0.008, 0.38), box_w - 0.016, 0.42,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor="gray", linewidth=0.6,
                transform=ax_bar.transAxes,
            )
            ax_bar.add_patch(rect)
            bright = 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
            ax_bar.text(x_c, 0.61, word, ha="center", va="center",
                        fontsize=10, fontweight="500",
                        color="black" if bright > 0.5 else "white",
                        transform=ax_bar.transAxes)
            ax_bar.text(x_c, 0.24, f"{score:.3f}", ha="center", va="center",
                        fontsize=8, color="#444", transform=ax_bar.transAxes)
        fig.tight_layout()
        return fig

    # ── POS-view plots ────────────────────────────────────────

    def plot_pos_overlay(self, alpha=0.6, figsize=None):
        """
        One overlay panel per POS category.
        Heatmaps of all words in that category are averaged together.
        Each panel uses the category's own color instead of inferno.
        """
        wpl = self._word_pos_list()

        # Group word_indices by broad POS
        groups = defaultdict(list)
        for (w, s, broad), idx in zip(wpl, self.word_indices):
            groups[broad].append(idx)

        # Sort categories by total attention score
        cat_scores = {cat: sum(self.heatmaps[i].mean() for i in idxs)
                      for cat, idxs in groups.items()}
        sorted_cats = sorted(cat_scores, key=lambda c: -cat_scores[c])

        n       = len(sorted_cats) + 1
        ncols   = min(4, n)
        nrows   = (n + ncols - 1) // ncols
        figsize = figsize or (4 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes      = np.array(axes).flatten()
        img512    = np.array(self.image.resize((512, 512)))

        axes[0].imshow(img512)
        axes[0].set_title("Generated image", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        for panel, cat in enumerate(sorted_cats, start=1):
            ax      = axes[panel]
            idxs    = groups[cat]
            # Average all heatmaps in this category
            combined = np.mean([self.heatmaps[i] for i in idxs], axis=0)
            combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

            # Build a colormap: transparent → category color
            hex_col  = _POS_COLORS.get(cat, '#94A3B8')
            rgba     = mcolors.to_rgba(hex_col)
            pos_cmap = LinearSegmentedColormap.from_list('pos', [(1,1,1,0), rgba[:3]])

            heat    = Image.fromarray((combined * 255).astype(np.uint8)) \
                           .resize((512, 512), Image.BILINEAR)
            colored = pos_cmap(np.array(heat) / 255.0)[..., :3]
            blended = np.clip((1 - alpha) * img512 / 255.0 + alpha * colored, 0, 1)

            # Words in this category
            cat_words = [w for w, _, b in wpl if b == cat]
            ax.imshow(blended)
            ax.set_title(cat, fontsize=10, fontweight="bold", color=hex_col)
            ax.set_xlabel(", ".join(cat_words), fontsize=7.5, color="#666")
            ax.axis("off")

        for ax in axes[panel + 1:]:
            ax.axis("off")

        fig.suptitle("Part-of-Speech attention overlay", fontsize=12, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_pos_heatmap(self, figsize=(14, 7)):
        """
        Two-panel POS visualization:
        ① All words colored by POS category (top panel)
        ② Bar chart of total attention per POS category (bottom panel)
        """
        wpl     = self._word_pos_list()
        totals  = self.pos_totals()

        fig     = plt.figure(figsize=figsize, facecolor='white')
        fig.suptitle("Part-of-Speech attention breakdown",
                     fontsize=12, fontweight="bold", y=0.98)

        # ── ① Word boxes colored by POS ──────────────────────
        ax_w = fig.add_axes([0.03, 0.44, 0.94, 0.50])
        ax_w.set_xlim(0, 1); ax_w.set_ylim(0, 1); ax_w.axis("off")
        ax_w.text(0, 0.97, "Words colored by part of speech  (opacity ∝ attention score)",
                  transform=ax_w.transAxes, fontsize=8,
                  color="#6B6B80", fontweight="600", va="top")

        n     = len(wpl)
        box_w = 1.0 / n
        max_s = max(s for _, s, _ in wpl) or 1.0

        for i, (word, score, broad) in enumerate(wpl):
            x_c     = (i + 0.5) * box_w
            hex_col = _POS_COLORS.get(broad, '#94A3B8')
            base    = mcolors.to_rgba(hex_col)
            # Opacity proportional to score (0.25 → 1.0)
            a       = 0.25 + 0.75 * (score / max_s)
            color   = (*base[:3], a)

            rect = mpatches.FancyBboxPatch(
                (i * box_w + 0.006, 0.32), box_w - 0.012, 0.46,
                boxstyle="round,pad=0.02",
                facecolor=color, edgecolor=hex_col, linewidth=0.8,
                transform=ax_w.transAxes,
            )
            ax_w.add_patch(rect)

            bright = 0.299*base[0] + 0.587*base[1] + 0.114*base[2]
            tc     = "white" if bright < 0.55 else "black"
            ax_w.text(x_c, 0.57, word, ha="center", va="center",
                      fontsize=8.5, color=tc, fontweight="500",
                      transform=ax_w.transAxes)
            ax_w.text(x_c, 0.20, f"{score:.3f}", ha="center", va="center",
                      fontsize=7, color="#555", transform=ax_w.transAxes)
            # Tiny POS label below score
            ax_w.text(x_c, 0.07, broad[:4], ha="center", va="center",
                      fontsize=6, color=hex_col, transform=ax_w.transAxes)

        # Legend
        seen_cats = list(dict.fromkeys(b for _, _, b in wpl))
        lx = 0.0
        for cat in seen_cats:
            c = _POS_COLORS.get(cat, '#94A3B8')
            ax_w.add_patch(mpatches.Rectangle(
                (lx, 0.88), 0.013, 0.07,
                facecolor=c, transform=ax_w.transAxes, zorder=5,
            ))
            ax_w.text(lx + 0.016, 0.915, cat,
                      transform=ax_w.transAxes,
                      fontsize=7, color="#444", va="center")
            lx += max(0.10, len(cat) * 0.012 + 0.02)

        # ── ② Bar chart of POS totals ─────────────────────────
        ax_b = fig.add_axes([0.08, 0.06, 0.88, 0.32])

        cats   = list(totals.keys())
        values = [totals[c]['total'] for c in cats]
        colors = [_POS_COLORS.get(c, '#94A3B8') for c in cats]

        bars = ax_b.bar(cats, values, color=colors, width=0.55,
                        edgecolor='white', linewidth=0.5, zorder=2)
        ax_b.set_axisbelow(True)
        ax_b.yaxis.grid(True, linestyle='--', alpha=0.4, color='#ccc')
        ax_b.set_ylabel("Total attention score", fontsize=8.5, color="#555")
        ax_b.tick_params(axis='x', colors='#333', labelsize=9)
        ax_b.tick_params(axis='y', colors='#888', labelsize=7.5)
        for spine in ['top', 'right']:
            ax_b.spines[spine].set_visible(False)
        for spine in ['left', 'bottom']:
            ax_b.spines[spine].set_color('#ddd')

        for bar, val in zip(bars, values):
            ax_b.text(bar.get_x() + bar.get_width()/2, val + 0.002,
                      f"{val:.3f}", ha="center", va="bottom",
                      fontsize=8, color="#555", fontfamily="monospace")

        fig.patch.set_facecolor('white')
        return fig

    # ── PIL helpers (for Gradio) ──────────────────────────────

    def _fig_to_pil(self, fig) -> Image.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def overlay_pil(self, alpha=0.55, top_k=10):
        return self._fig_to_pil(self.plot_overlay(alpha=alpha, top_k=top_k))

    def text_heatmap_pil(self, top_k=10):
        return self._fig_to_pil(self.plot_text_heatmap(top_k=top_k))

    def pos_overlay_pil(self, alpha=0.6):
        return self._fig_to_pil(self.plot_pos_overlay(alpha=alpha))

    def pos_heatmap_pil(self):
        return self._fig_to_pil(self.plot_pos_heatmap())

    # ── notebook convenience ──────────────────────────────────

    def show(self, top_k=10):
        self.plot_overlay(top_k=top_k).show()
        self.plot_text_heatmap(top_k=top_k).show()

    def show_pos(self):
        self.plot_pos_overlay().show()
        self.plot_pos_heatmap().show()

    def save(self, folder=".", prefix=None, top_k=10):
        folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
        tag    = prefix or self.prompt[:30].replace(" ", "_")
        paths  = {
            "overlay"      : folder / f"{tag}_overlay.png",
            "text_heatmap" : folder / f"{tag}_text_heatmap.png",
            "pos_overlay"  : folder / f"{tag}_pos_overlay.png",
            "pos_heatmap"  : folder / f"{tag}_pos_heatmap.png",
        }
        self.plot_overlay(top_k=top_k)  .savefig(paths["overlay"],       dpi=130, bbox_inches="tight")
        self.plot_text_heatmap(top_k=top_k).savefig(paths["text_heatmap"],dpi=130, bbox_inches="tight")
        self.plot_pos_overlay()          .savefig(paths["pos_overlay"],   dpi=130, bbox_inches="tight")
        self.plot_pos_heatmap()          .savefig(paths["pos_heatmap"],   dpi=130, bbox_inches="tight")
        plt.close("all")
        for k, p in paths.items():
            print(f"  {k}: {p}")
        return paths



class HeatmapGenerator:
    """
    Load once, run many prompts.

    gen = HeatmapGenerator()                          # SD 1.5
    r   = gen.run("a golden retriever in the snow")
    r.show()        # word view
    r.show_pos()    # POS view
    r.save("out/")

    gen.switch_model("stabilityai/stable-diffusion-xl-base-1.0")
    r2  = gen.run("same prompt, better quality")
    """

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5",
                 device=None, steps=30, capture_n=10):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.steps     = steps
        self.capture_n = capture_n
        self.model_id  = None
        self.pipe      = None
        self.is_xl     = False
        self._load(model_id)

    def _load(self, model_id: str):
        if self.pipe is not None:
            print(f"Unloading {self.model_id}…")
            del self.pipe; self.pipe = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

        is_xl  = any(x in model_id.lower() for x in ["xl", "sdxl", "stable-diffusion-xl"])
        Cls    = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
        kwargs = dict(torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                      use_safetensors=True)
        if not is_xl:
            kwargs["safety_checker"]          = None
            kwargs["requires_safety_checker"] = False

        print(f"Loading {'SDXL' if is_xl else 'SD'}: {model_id}…")
        self.pipe  = Cls.from_pretrained(model_id, **kwargs)
        self.is_xl = is_xl

        if is_xl:
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.to(torch.float32)
        else:
            self.pipe = self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()

        self.model_id = model_id
        print(f"Ready — {model_id}")

    def switch_model(self, model_id: str):
        if model_id != self.model_id:
            self._load(model_id)

    def run(self, prompt: str, seed=42, steps=None, guidance=7.5, capture_n=None):
        steps     = steps     or self.steps
        capture_n = capture_n or self.capture_n

        tokenizer  = self.pipe.tokenizer
        tokens     = tokenizer.encode(prompt)
        token_strs = tokenizer.convert_ids_to_tokens(tokens)

        store = _AttentionStore(capture_last_n=capture_n, total_steps=steps)
        store.register(self.pipe.unet)

        generator = torch.Generator(device="cpu").manual_seed(seed)
        out       = self.pipe(prompt, num_inference_steps=steps,
                              guidance_scale=guidance, generator=generator,
                              callback_on_step_end=store.on_step_end)
        image    = out.images[0]
        heatmaps = store.aggregate(len(tokens))
        store.restore(self.pipe.unet)

        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return HeatmapResult(image=image, tokens=token_strs,
                             heatmaps=heatmaps, prompt=prompt)
