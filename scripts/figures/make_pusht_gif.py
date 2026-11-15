# Author: Mohammad Al-Jarrah
"""
The animated Push-T figure of the README (assets/pusht_rollouts.gif): the eight policies of
Table 2 in a 2 x 4 layout, each a 3 x 3 grid of the same nine held-out episodes (seed-0
checkpoints). Above each grid: the policy, its mean score +- one standard error over the 1,500
episodes of Table 2, and its time per episode. In each episode: the score so far, and, once the
episode ends, a green frame for a completed task (score 1) or a red frame for a failure (score
below 0.5).

    python scripts/figures/make_pusht_gif.py                 # needs checkpoints/pusht/<policy>_seed0.pt
    GRID=4 python scripts/figures/make_pusht_gif.py          # 4 x 4 episodes per policy

The rollouts are cached in results/figures/; delete the cache to simulate again.
"""
import os
import sys

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
from PIL import Image, ImageDraw

import experiment as X
from visualization import grid_frame, _font

# (label, policy, CP steps) and the statistics of Table 2 shown in the titles: score, SE, s/episode
POLICIES = [('OTP', 'otp', 1), ('FMP', 'fmp', 1), ('SIP', 'sip', 1), ('KRP', 'krp', 1),
            ('DP', 'dp', 1), ('DPR', 'dpr', 1), ('CP 1-step', 'cp', 1), ('CP 3-step', 'cp', 3)]
TABLE2 = {'OTP': (0.943, 0.005, 0.12), 'FMP': (0.947, 0.004, 3.74), 'SIP': (0.965, 0.004, 2.10),
          'KRP': (0.914, 0.006, 1.25), 'DP': (0.915, 0.007, 5.40), 'DPR': (0.963, 0.004, 2.84),
          'CP 1-step': (0.938, 0.006, 0.17), 'CP 3-step': (0.940, 0.006, 0.29)}

GRID     = int(os.environ.get('GRID', 3))             # GRID x GRID episodes per policy
PER_ROW  = 4                                          # policies per row
STRIDE   = 2                                          # show every second step (2x real time at FPS 10)
FPS      = 10
HOLD     = 20                                         # frames holding the final state
GAP, TITLE_H, FOOTER_H, MARGIN = 18, 34, 26, 10
OUT      = os.path.join(ROOT, 'assets', 'pusht_rollouts.gif')
CACHE    = os.path.join(ROOT, 'results', 'figures', 'pusht_gif_rollouts_grid%d.npz' % GRID)


def episodes():
    """GRID**2 held-out episode seeds drawn uniformly from 10000-10499 (the episodes of Figure 4 for GRID = 3)."""
    return [10000 + int(i) for i in np.sort(np.random.default_rng(0).choice(500, GRID * GRID, replace=False))]


def ragged(arrays):
    """A 1-D object array holding arrays of different lengths (the episodes end at different steps)."""
    out = np.empty(len(arrays), dtype=object)
    for k, a in enumerate(arrays):
        out[k] = a
    return out


def simulate():
    """Rendered frames and per-step rewards of every policy on the episodes, cached on disk."""
    if os.path.exists(CACHE):
        return dict(np.load(CACHE, allow_pickle=True))
    data = {}
    for label, policy, steps in POLICIES:
        agent, cfg = X.make_agent('pusht', policy, X.checkpoint_path('pusht', policy, 0), cp_steps=steps)
        res = X.run_episodes('pusht', agent, cfg, episodes(), render=True)
        data[label + '/frames'] = ragged([np.asarray(f, np.uint8) for f in res['frames']])
        data[label + '/running'] = ragged([np.maximum.accumulate(r) for r in res['rewards']])
        data[label + '/score'] = np.asarray(res['score'])
        print('%-10s %s' % (label, np.round(res['score'], 2)), flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **data)
    return data


def main():
    data = simulate()
    labels = [p[0] for p in POLICIES]
    panel = data[labels[0] + '/frames'][0].shape[1]
    block = GRID * panel + (GRID - 1) * 3
    rows = int(np.ceil(len(labels) / PER_ROW))
    W = 2 * MARGIN + PER_ROW * block + (PER_ROW - 1) * GAP
    H = MARGIN + rows * (TITLE_H + block) + (rows - 1) * GAP + FOOTER_H
    base = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(base)
    origin = {}
    for k, label in enumerate(labels):
        r, c = divmod(k, PER_ROW)
        x0, y0 = MARGIN + c * (block + GAP), MARGIN + r * (TITLE_H + block + GAP)
        score, se, sec = TABLE2[label]
        d.text((x0 + block / 2, y0), label, font=_font(15), fill=(20, 20, 20), anchor='mt')
        d.text((x0 + block / 2, y0 + 18), 'score %.3f ± %.3f, %.2f s/episode' % (score, se, sec),
               font=_font(12, bold=False), fill=(90, 90, 90), anchor='mt')
        origin[label] = (x0, y0 + TITLE_H)
    d.text((MARGIN, H - FOOTER_H + 6), 'Same %d held-out initial conditions for every policy. Frame: green = task '
           'completed (score 1), red = failure (score < 0.5). Titles: mean ± SE and time per episode over '
           '1,500 episodes.' % (GRID * GRID), font=_font(12, bold=False), fill=(90, 90, 90))
    T = max(len(f) for label in labels for f in data[label + '/frames'])
    frames = []
    for t in list(range(0, T, STRIDE)) + [T - 1] * HOLD:
        img = base.copy()
        for label in labels:
            eps = data[label + '/frames']
            idx = [min(t, len(f) - 1) for f in eps]
            done = [t >= len(f) - 1 for f in eps]
            scores = [data[label + '/running'][k][i] for k, i in enumerate(idx)]   # score so far
            img.paste(grid_frame([f[i] for f, i in zip(eps, idx)], scores, done, GRID), origin[label])
        ImageDraw.Draw(img).text((W - MARGIN, H - FOOTER_H + 6), 'step %d' % min(t + 1, T),
                                 font=_font(12, bold=False), fill=(90, 90, 90), anchor='ra')
        frames.append(img.convert('P', palette=Image.ADAPTIVE, colors=128))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames[0].save(OUT, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print('saved %s: %d x %d px, %d frames, %.1f MB' % (OUT, W, H, len(frames), os.path.getsize(OUT) / 2 ** 20))


if __name__ == '__main__':
    main()
