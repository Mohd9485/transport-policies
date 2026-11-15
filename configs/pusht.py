# Author: Mohammad Al-Jarrah
"""
Hyperparameters of every policy on Push-T: the settings that produced Table 2 of the paper.

Horizons: h_o observation steps are given to the policy, it predicts h_p actions, and the first
h_a of them are executed before the next query. The settings of the proposed policies and of
DPR and KRP were selected with SMAC3 (tuning/search_spaces.py, whose defaults are these values);
DP and CP use the published settings of their authors.
"""

CONFIGS = {
    # ---------------------------------------------------------------- proposed policies (reference P_A)
    'otp': dict(
        h_o=1, h_p=4, h_a=4,
        width=[576, 192], n_blocks=[3, 3],              # [critic, map]
        lr=[5.97116761e-5, 7.231071011e-4],             # [critic, map]
        batch_size=576, iterations=16384, k_in=25,      # k_in map steps per critic step
        weight_decay=0.0, grad_clip=0.25, warmup_frac=0.05, act='mish', ema_decay=0.0,
    ),
    'fmp': dict(
        h_o=1, h_p=4, h_a=4,
        width=896, n_blocks=5, time_embed_dim=20, time_scale=100.0, time_max_period=50.0, act='mish',
        lr=9.329118057e-4, batch_size=256, iterations=16384,
        weight_decay=0.2, grad_clip=1.0, warmup_frac=0.02, ema_decay=0.0,
        n_steps=100,                                    # Heun steps at deployment
    ),
    'sip': dict(
        h_o=1, h_p=8, h_a=8,
        width=512, n_blocks=5, time_embed_dim=10, time_scale=150.0, time_max_period=50.0, act='relu',
        noise_level=0.1146888409648, denoiser_weight=1.0,
        lr=1.9173090338e-3, batch_size=320, iterations=65536,
        weight_decay=0.01, grad_clip=0.25, warmup_frac=0.02, ema_decay=0.0,
        n_steps=100,                                    # stochastic Heun steps at deployment
    ),
    # ---------------------------------------------------------------- Gaussian-reference policies
    'krp': dict(
        h_o=1, h_p=8, h_a=8,
        width=320, depth=3, act='tanh', quad_points=16, residual=False,
        lr=2.8776630053e-3, batch_size=320, iterations=32768,
        weight_decay=1e-3, grad_clip=5.0, warmup_frac=0.05, ema_decay=0.0,
    ),
    'dp': dict(                                         # Chi et al. (2023), state-based Push-T
        h_o=2, h_p=16, h_a=8, act_dim=2,
        step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8,
        n_train_timesteps=100, n_inference_steps=100,
        lr=1e-4, betas=[0.95, 0.999], weight_decay=1e-6, warmup_steps=500,
        batch_size=256, iterations=505000,              # 5,000 epochs of the released configuration
    ),
    'dpr': dict(
        h_o=1, h_p=8, h_a=8,
        width=1024, n_blocks=4, time_embed_dim=10, time_scale=150.0, time_max_period=50.0, act='mish',
        lr=3.683678514e-4, batch_size=256, iterations=65536,
        weight_decay=0.1, grad_clip=0.1, warmup_frac=0.02,
        n_steps=100,
    ),
    'cp': dict(                                         # Prasad et al. (2024), adapted to states
        h_o=2, h_p=16, h_a=8, act_dim=2,
        step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8, dropout=0.2,
        lr=1e-4, betas=[0.95, 0.999], weight_decay=1e-6, warmup_steps=500, batch_size=256,
        iterations_teacher=150000, iterations_student=150000, lr_schedule_student=120000,
        chaining_times=[27.0, 54.0],                    # noise levels of the two extra steps of CP 3-step
    ),
}
