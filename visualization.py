# Author: Mohammad Al-Jarrah
"""
Plots and animations of rollouts, used by the notebooks and scripts/figures/.

    grid_gif(episode_frames, labels, path)        Push-T: animated grid of rendered episodes
    final_frames(episode_frames, scores)          Push-T: the last frame of every episode, in a grid
    pusht_chunks(policy_agent, obs_history, ...)  Push-T: sampled action chunks for one observation
    square_scenes(results, ...)                   Square: top-down start and end poses of the nut
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

GREEN, RED = (61, 140, 92), (184, 66, 58)


def _font(size, bold=True):
    """DejaVu Sans at the given size, or the default font of PIL."""
    path = '/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf' % ('-Bold' if bold else '')
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def grid_frame(images, scores, done, grid, gap=3, success=None):
    """
    One frame of a grid of episodes: images (n, H, W, 3), the score so far of each episode, and
    whether it has ended. An ended episode gets a green frame if it completed the task (score 1)
    and a red frame if it failed (score below 0.5).
    """
    H, W = images[0].shape[:2]
    rows = int(np.ceil(len(images) / grid))
    canvas = Image.new('RGB', (grid * W + (grid - 1) * gap, rows * H + (rows - 1) * gap), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _font(max(9, W // 9))
    for k, img in enumerate(images):
        x, y = (k % grid) * (W + gap), (k // grid) * (H + gap)
        canvas.paste(Image.fromarray(img), (x, y))
        s = np.floor(scores[k] * 100) / 100                     # truncated: 1.00 means completed
        if done[k] and (s >= 1.0 or s < 0.5):
            col = GREEN if s >= 1.0 else RED
            for b in range(3):
                draw.rectangle([x + b, y + b, x + W - 1 - b, y + H - 1 - b], outline=col)
        txt = '%.2f' % s
        tw = draw.textlength(txt, font=font)
        draw.rectangle([x + W - tw - 8, y + 4, x + W - 4, y + 5 + font.size], fill=(255, 255, 255))
        draw.text((x + W - 6, y + 5), txt, font=font, fill=(20, 20, 20), anchor='ra')
    return canvas


def grid_gif(episode_frames, episode_rewards, path, grid=3, stride=2, fps=10, title=None, hold=20):
    """
    Animated grid of Push-T episodes. episode_frames: list of (T_e, H, W, 3) arrays;
    episode_rewards: list of per-step rewards (the score so far is their running maximum).
    A finished episode holds its last frame. Every `stride`-th step is shown.
    """
    T = max(len(f) for f in episode_frames)
    running = [np.maximum.accumulate(np.asarray(r)) for r in episode_rewards]
    frames = []
    for t in list(range(0, T, stride)) + [T - 1] * hold:
        idx = [min(t, len(f) - 1) for f in episode_frames]
        img = grid_frame([f[i] for f, i in zip(episode_frames, idx)], [r[i] for r, i in zip(running, idx)],
                         [t >= len(f) - 1 for f in episode_frames], grid)
        if title:
            top = Image.new('RGB', (img.width, 24), (255, 255, 255))
            ImageDraw.Draw(top).text((img.width / 2, 5), title, font=_font(13), fill=(20, 20, 20), anchor='mt')
            full = Image.new('RGB', (img.width, img.height + 24), (255, 255, 255))
            full.paste(top, (0, 0)); full.paste(img, (0, 24))
            img = full
        frames.append(img.convert('P', palette=Image.ADAPTIVE, colors=128))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return path


def final_frames(episode_frames, scores, grid=3):
    """The last frame of every episode, with its score and outcome frame (a PIL image)."""
    return grid_frame([f[-1] for f in episode_frames], scores, [True] * len(scores), grid)


def pusht_chunks(agent, obs_history, frame, n_samples=64, ax=None):
    """
    The action chunks that the policy samples for one observation history, drawn on the rendered
    frame of the environment (Push-T coordinates 0-512 are mapped to the frame's pixels).
    """
    import torch
    ax = ax or plt.gca()
    s = agent.normalize(np.asarray(obs_history, np.float32).reshape(1, -1), agent.stats['S'])
    z = agent.normalize(agent.A_pool[agent.rng.integers(0, len(agent.A_pool), n_samples)], agent.stats['A'])
    with torch.no_grad():
        y = agent.policy.sample(torch.as_tensor(s, device=agent.device).repeat(n_samples, 1),
                                torch.as_tensor(np.clip(z, -1, 1), device=agent.device)).cpu().numpy()
    chunks = agent.denormalize(y, agent.stats['A']).reshape(n_samples, -1, 2)
    scale = frame.shape[0] / 512.0
    ax.imshow(frame)
    for c in chunks:
        ax.plot(c[:, 0] * scale, c[:, 1] * scale, color='C0', lw=0.8, alpha=0.4)
        ax.scatter(c[0, 0] * scale, c[0, 1] * scale, s=6, color='C3', zorder=3)
    ax.set_xticks([]); ax.set_yticks([])
    return chunks


# ---------------------------------------------------------------- Square
PEG_XY, PEG_HALF = np.array([0.23, 0.10]), 0.016        # the square peg of NutAssemblySquare
RING_OUT, RING_IN, HANDLE = 0.0437, 0.0227, (0.0437, 0.0793, 0.0159)


def _nut(obs):
    """Top-down outline of the nut (outer square, inner square, handle) from an observation."""
    x, y, z, w = obs[3:7]                                   # quaternion in robosuite's (x, y, z, w) order
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    sq = lambda h: np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    x0, x1, hy = HANDLE
    return [p @ R.T + obs[:2] for p in (sq(RING_OUT), sq(RING_IN), np.array([[x0, -hy], [x1, -hy], [x1, hy], [x0, hy]]))]


def square_scenes(start_obs, end_obs, success, cols=4, size=1.6):
    """Top-down scenes of Square episodes: peg (green), nut at the start (gray) and end (dashed)."""
    from matplotlib.patches import Polygon
    n = len(start_obs)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if k >= n:
            continue
        ax.add_patch(Polygon(PEG_XY + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * PEG_HALF,
                             color='#5bb37f', alpha=0.6))
        out, inn, han = _nut(start_obs[k])
        for p in (out, han):
            ax.add_patch(Polygon(p, facecolor='#7d7c76', edgecolor='#52514e', alpha=0.55))
        ax.add_patch(Polygon(inn, facecolor='white', edgecolor='#52514e'))
        for p in _nut(end_obs[k]):
            ax.add_patch(Polygon(p, facecolor='none', edgecolor='#52514e', ls='--'))
        ax.scatter(*start_obs[k][14:16], s=12, facecolor='none', edgecolor='k')    # end effector
        ax.set_title('success' if success[k] else 'failure', fontsize=8,
                     color='#3d8c5c' if success[k] else '#b8423a')
        ax.set_xlim(-0.3, 0.33); ax.set_ylim(-0.15, 0.33); ax.set_aspect('equal')
    fig.tight_layout()
    return fig
