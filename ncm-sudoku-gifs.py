import argparse
import os
import pickle
import runpy
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ----------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------
class SudokuRenderer:
    """Draws 9x9 boards in a small warm palette (cream, sand, navy)."""

    CELL, PAD, HEADER, FOOTER = 60, 24, 56, 76
    INK, CREAM = (16, 44, 87), (248, 240, 229)          # #102C57, #F8F0E5
    SAND, TAN = (234, 219, 200), (218, 192, 163)        # #EADBC8, #DAC0A3
    WHITE, CHANGE = (255, 255, 255), (195, 213, 240)
    _NUNITO_VARIABLE = ("Nunito[wght].ttf", "Nunito-VariableFont_wght.ttf")

    def __init__(self, level_seconds=2.0, final_seconds=5.0, lock_clues=False):
        self.level_seconds = level_seconds
        self.final_seconds = final_seconds
        self.lock_clues = lock_clues
        v = self._NUNITO_VARIABLE
        # Uses Nunito if it is installed (weights from the CSS: 200 / 400 / 700), else a common sans-serif.
        self.title_font = self._font(("Nunito-Light.ttf", "Nunito-ExtraLight.ttf") + v
                                     + ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"), 26, 200)
        self.small_font = self._font(("Nunito-Regular.ttf",) + v
                                     + ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"), 14, 400)
        self.digit_font = self._font(("Nunito-Bold.ttf",) + v
                                     + ("DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"), 32, 700)

    @staticmethod
    def _font(names, size, weight=None):
        for name in names:
            try:
                font = ImageFont.truetype(name, size)
            except OSError:
                continue
            if weight is not None:
                try:
                    font.set_variation_by_axes([weight])
                except Exception:
                    pass
            return font
        try:
            return ImageFont.load_default(size)
        except TypeError:
            return ImageFont.load_default()

    def _shown(self, grid, givens):
        grid = np.asarray(grid).reshape(9, 9)
        return np.where(givens != 0, givens, grid) if self.lock_clues else grid

    def render(self, grid, givens, prev=None, title=""):
        givens = np.asarray(givens).reshape(9, 9)
        shown = self._shown(grid, givens)
        prev = None if prev is None else self._shown(prev, givens)
        C, P, H = self.CELL, self.PAD, self.HEADER

        img = Image.new("RGB", (9 * C + 2 * P, H + 9 * C + self.FOOTER), self.WHITE)
        d = ImageDraw.Draw(img)
        d.text((P, 14), title, fill=self.INK, font=self.title_font)

        for r in range(9):
            for c in range(9):
                x0, y0, v = P + c * C, H + r * C, int(shown[r, c])
                frame = self.SAND if givens[r, c] != 0 else self.WHITE
                d.rectangle([x0, y0, x0 + C, y0 + C], fill=frame, outline=self.TAN)
                if prev is not None and v != prev[r, c]:
                    d.rectangle([x0 + 7, y0 + 7, x0 + C - 7, y0 + C - 7], fill=self.CHANGE)
                if givens[r, c] != 0 and v != givens[r, c]:
                    d.text((x0 + 9, y0 + 8), str(int(givens[r, c])), fill=self.INK,
                           font=self.small_font, anchor="la")
                if v != 0:
                    d.text((x0 + C // 2, y0 + C // 2), str(v), fill=self.INK,
                           font=self.digit_font, anchor="mm")

        for i in range(0, 10, 3):                                     # 3x3 box borders
            d.line([P + i * C, H, P + i * C, H + 9 * C], fill=self.INK, width=3)
            d.line([P, H + i * C, P + 9 * C, H + i * C], fill=self.INK, width=3)

        self._legend(d, P, H + 9 * C + 14)
        return img

    def _swatch(self, d, x, y, frame, changed=False, size=26, inset=5):
        d.rectangle([x, y, x + size, y + size], fill=frame, outline=self.TAN)
        if changed:
            d.rectangle([x + inset, y + inset, x + size - inset, y + size - inset], fill=self.CHANGE)

    def _legend(self, d, x, y):
        x_start, size = x, 26
        items = (("Clue", [(self.SAND, False)]),
                 ("Guess", [(self.WHITE, False)]),
                 ("Changed", [(self.SAND, True), (self.WHITE, True)]))
        for label, swatches in items:
            for frame, changed in swatches:
                self._swatch(d, x, y, frame, changed, size)
                x += size + 4
            x += 4
            d.text((x, y + size // 2), label, fill=self.INK, font=self.small_font, anchor="lm")
            x += int(d.textlength(label, font=self.small_font)) + 26
        if not self.lock_clues:
            d.text((x_start, y + size + 10), "Small digit is the real clue",
                   fill=self.INK, font=self.small_font, anchor="la")

    def save_gif(self, level_grids, givens, path):
        n, prev, frames = len(level_grids), None, []
        for k, g in enumerate(level_grids, 1):
            title = f"Level {k} of {n}"
            frames.append(self.render(g, givens, prev, title))
            prev = g
        ms = [int(self.level_seconds * 1000)] * (n - 1) + [int(self.final_seconds * 1000)]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0)


class LevelCapture:
    """Wraps VQVAETokenizer.from_config so that, once the repo script builds its
    tokenizer, the VQ's idx_to_zhats(..., return_zlevels=True) also decodes every
    level and stores the argmax token ids. The script's own behaviour is unchanged."""

    def __init__(self):
        self.tokenizer = None
        self.calls = []
        self._orig = None

    def install(self):
        from tokenization.vqvae_tokenizer import VQVAETokenizer
        self._cls = VQVAETokenizer
        self._orig = VQVAETokenizer.from_config
        capture = self

        def from_config(*args, **kw):
            tk = capture._orig(*args, **kw)
            capture._hook(tk)
            return tk

        VQVAETokenizer.from_config = staticmethod(from_config)

    def _hook(self, tk):
        import torch
        self.tokenizer = tk
        vq = tk.vqvae.vq
        original = vq.idx_to_zhats
        calls = self.calls

        def idx_to_zhats(*a, **kw):
            out = original(*a, **kw)
            if kw.get("return_zlevels"):
                with torch.no_grad():
                    calls.append([tk.vqvae.decoder(z).argmax(-1).cpu() for z in out[1]])
            return out

        vq.idx_to_zhats = idx_to_zhats

    def uninstall(self):
        if self._orig is not None:
            self._cls.from_config = staticmethod(self._orig)

    def per_puzzle(self):
        return [[lvl[b].tolist() for lvl in call]
                for call in self.calls for b in range(call[0].shape[0])]


class SudokuLevelGifs:
    DIFFICULTIES = ("easy", "medium", "hard")

    def __init__(self, ncm_root, difficulty, n=10, bs=8, seed=0, level_seconds=2.0,
                 final_seconds=5.0, lock_clues=False, only_solved=False, out_dir="gifs",
                 checkpoint_dir="", data_dir="", split="", extra_args=()):
        self.root = os.path.abspath(ncm_root)
        self.difficulty, self.task = difficulty, f"sudoku-{difficulty}"
        self.script = os.path.join(self.root, "scripts", "eval_shard.py")
        self.config = ""
        self._checkpoint_dir = os.path.abspath(checkpoint_dir) if checkpoint_dir else ""
        self._data_dir = os.path.abspath(data_dir) if data_dir else ""
        self.split, self.extra_args = split, list(extra_args)
        self.n, self.bs, self.seed = n, bs, seed
        self.only_solved, self.out_dir = only_solved, os.path.abspath(out_dir)
        self.renderer = SudokuRenderer(level_seconds, final_seconds, lock_clues)
        self.capture = LevelCapture()

    @staticmethod
    def _resolve(kind, explicit, candidates, files, download_cmd):
        for d in ([explicit] if explicit else candidates):
            if all(os.path.isfile(os.path.join(d, f)) for f in files):
                return d
        tried = "\n    ".join([explicit] if explicit else candidates)
        sys.exit(f"Could not find the {kind} files {files} in:\n    {tried}\n"
                 f"  Download them from inside the ncm folder with:\n    {download_cmd}\n"
                 f"  or point to them with --{kind}-dir.")

    def _check_layout(self):
        from omegaconf import OmegaConf
        if not os.path.isdir(self.root):
            sys.exit(f"NCM repo not found at {self.root}\n"
                     f"  Clone it next to this repo (git clone https://github.com/HazyResearch/ncm.git) "
                     f"or pass --ncm-root.")
        j = lambda *p: os.path.join(self.root, *p)
        if not os.path.isfile(self.script):
            sys.exit(f"Missing scripts/eval_shard.py in {self.root}. Is it a full clone of HazyResearch/ncm?")

        configs = [j("config", "eval", self.task + ext) for ext in (".yml", ".yaml")]
        self.config = next((p for p in configs if os.path.isfile(p)), "")
        if not self.config:
            tried = "\n    ".join(configs)
            sys.exit(f"Could not find a config for {self.task} in:\n    {tried}\n"
                     f"  Is it a full clone of HazyResearch/ncm?")
        self.cfg = OmegaConf.load(self.config)
        self.split = self.split or self.cfg.eval.get("split", "test")

        self.checkpoint_dir = self._resolve(
            "checkpoint", self._checkpoint_dir,
            [j("artifacts", "sudoku", self.difficulty), j("artifacts", self.task)],
            [self.cfg.checkpoints.tokenizer, self.cfg.checkpoints.generator],
            "hf download hazyresearch/ncm-checkpoints --local-dir artifacts")
        self.data_dir = self._resolve(
            "data", self._data_dir,
            [j("data", "sudoku", self.difficulty), j("data", self.task), j("data", "sudoku")],
            [self.cfg.data[self.split], self.cfg.data[f"{self.split}_refs"]],
            "hf download hazyresearch/ncm-datasets --repo-type dataset --local-dir data")

        rel = lambda p: os.path.relpath(p, self.root) if p.startswith(self.root) else p
        print(f"NCM clone   : {self.root}")
        print(f"  config    : {rel(self.config)}")
        print(f"  checkpoint: {rel(self.checkpoint_dir)}")
        print(f"  data      : {rel(self.data_dir)}")

    def run_script(self):
        import torch
        sys.dont_write_bytecode = True
        sys.path.insert(0, self.root)
        os.environ["NCM_ROOT"] = self.root
        argv = ["--config", self.config, "--checkpoint-dir", self.checkpoint_dir,
                "--data-dir", self.data_dir, "--split", self.split]
        argv += self.extra_args + ["--n", str(self.n), "--bs", str(self.bs), "--out", self.pkl]

        torch.manual_seed(self.seed)
        self.capture.install()
        cwd, old_argv = os.getcwd(), sys.argv
        os.chdir(self.root)
        sys.argv = [self.script] + argv
        try:
            runpy.run_path(self.script, run_name="__main__")
        finally:
            os.chdir(cwd)
            sys.argv = old_argv
            self.capture.uninstall()

    @staticmethod
    def to_board(text):
        chars = [c for c in text if c.isdigit() or c in "._-?*"]
        vals = [int(c) if c in "123456789" else 0 for c in chars]
        if len(vals) != 81:
            print(f"  WARNING: parsed {len(vals)} cells, expected 81. Raw text: {text!r}")
        return (vals + [0] * 81)[:81]

    def _decode(self, ids):
        pre = self.capture.tokenizer.pre_tokenizer
        eos, pad = pre.eos_token_id, pre.pad_token_id
        ids = list(ids)
        if eos in ids:
            ids = ids[:ids.index(eos)]
        return pre.decode([t for t in ids if t not in (eos, pad)], skip_special_tokens=True)

    def build_gifs(self):
        items = pickle.load(open(os.path.join(self.data_dir, self.cfg.data[self.split]), "rb"))
        P = int(self.cfg.model.prefix_len)
        pad = self.capture.tokenizer.pre_tokenizer.pad_token_id

        out = pickle.load(open(self.pkl, "rb"))
        levels = self.capture.per_puzzle()
        if len(levels) != len(out["hyps"]):
            sys.exit(f"Captured {len(levels)} puzzles but the script produced {len(out['hyps'])}; "
                     f"the hook probably fired on extra calls. Tell me and I'll adjust.")
        print(f"captured {len(levels[0])} levels per puzzle")

        written = []
        for i, idx in enumerate(out["idxs"]):
            solved = self.to_board(out["hyps"][i]) == self.to_board(out["refs"][i])
            if self.only_solved and not solved:
                continue
            givens = self.to_board(self._decode((list(items[idx])[:P] + [pad] * P)[:P]))
            grids = [self.to_board(self._decode(ids)) for ids in levels[i]]
            name = f"sudoku_{self.difficulty}_{idx:03d}_{'solved' if solved else 'unsolved'}.gif"
            path = os.path.join(self.out_dir, name)
            self.renderer.save_gif(grids, givens, path)
            print(f"puzzle {idx}: {'solved' if solved else 'NOT solved'} -> {path}")
            written.append(path)
        return written

    def run(self):
        self._check_layout()
        os.makedirs(self.out_dir, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.pkl = os.path.join(tmp, "shard.pkl")
            self.run_script()
            return self.build_gifs()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description="Per-level sudoku GIFs for NCM. Unknown flags are forwarded to NCM's eval_shard.py.")
    ap.add_argument("--difficulty", required=True, choices=SudokuLevelGifs.DIFFICULTIES)
    ap.add_argument("--n", type=int, default=10, help="use the first N test puzzles")
    ap.add_argument("--out-dir", default="gifs", help="where the GIFs go (relative to where you run this)")
    ap.add_argument("--ncm-root", default=os.environ.get("NCM_ROOT") or os.path.join(here, "..", "ncm"),
                    help="the NCM clone (default: ../ncm next to this repo, or $NCM_ROOT)")
    ap.add_argument("--checkpoint-dir", default="", help="override the checkpoint folder")
    ap.add_argument("--data-dir", default="", help="override the data folder")
    ap.add_argument("--split", default="")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--level-seconds", type=float, default=2.0, help="time on each level")
    ap.add_argument("--final-seconds", type=float, default=5.0, help="time on the final level")
    ap.add_argument("--lock-clues", action="store_true",
                    help="always draw the true clue digit (default: draw what the model decoded, "
                         "with the real clue as a small digit where it differs)")
    ap.add_argument("--only-solved", action="store_true")
    a, extra = ap.parse_known_args()

    SudokuLevelGifs(a.ncm_root, a.difficulty, n=a.n, bs=a.bs, seed=a.seed,
                    level_seconds=a.level_seconds, final_seconds=a.final_seconds,
                    lock_clues=a.lock_clues, only_solved=a.only_solved, out_dir=a.out_dir,
                    checkpoint_dir=a.checkpoint_dir, data_dir=a.data_dir, split=a.split,
                    extra_args=extra).run()


if __name__ == "__main__":
    main()
