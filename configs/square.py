# Author: Mohammad Al-Jarrah
"""
Hyperparameters of every policy on Square: the settings that produced Table 3 of the paper.

The two modifications of Table 3 are options of the scripts, not separate configurations:
    --state-noise   adds Gaussian noise with standard deviation STATE_NOISE (in normalized units of
                    the min-max normalization) to the object-pose dimensions of the conditioning
                    states during training; a separate checkpoint is trained;
    --mean-of N     samples N chunks for every query and executes their mean (N = 8 in the paper).

DP and CP use the normalization of Diffusion Policy's Square configuration ('dp'); the state
noise is converted to the same physical standard deviation in that normalization.
"""

STATE_NOISE = 0.02
MEAN_OF     = 8

CONFIGS = {
    # ---------------------------------------------------------------- proposed policies (reference P_A)
    'otp': dict(
        h_o=2, h_p=4, h_a=4,
        width=[896, 1344], n_blocks=[3, 3],             # [critic, map]
        lr=[6.80628782e-5, 2.483059228e-4],             # [critic, map]
        batch_size=1152, iterations=32768, k_in=25,
        weight_decay=0.0, grad_clip=1.0, warmup_frac=0.05, act='mish', ema_decay=0.999,
    ),
    'fmp': dict(
        h_o=2, h_p=4, h_a=4,
        width=1024, n_blocks=6, time_embed_dim=30, time_scale=150.0, time_max_period=50.0, act='mish',
        lr=3.624188125e-4, batch_size=1280, iterations=131072,
        weight_decay=1e-3, grad_clip=0.5, warmup_frac=0.05, ema_decay=0.9999,
        n_steps=100,
    ),
    'sip': dict(
        h_o=2, h_p=4, h_a=4,
        width=1408, n_blocks=6, time_embed_dim=40, time_scale=150.0, time_max_period=50.0, act='mish',
        noise_level=0.0868771063816, denoiser_weight=1.0,
        lr=5.440840501e-4, batch_size=1216, iterations=131072,
        weight_decay=0.01, grad_clip=1.0, warmup_frac=0.05, ema_decay=0.9999,
        n_steps=100,
    ),
    # ---------------------------------------------------------------- Gaussian-reference policies
    'krp': dict(
        h_o=2, h_p=4, h_a=4,
        width=320, depth=3, act='mish', quad_points=16, residual=True,
        lr=3e-3, batch_size=576, iterations=49152,
        weight_decay=0.01, grad_clip=1.0, warmup_frac=0.0, ema_decay=0.999,
    ),
    'dp': dict(                                         # Diffusion Policy, low-dimensional Square
        h_o=2, h_p=16, h_a=8, act_dim=10, normalization='dp',
        step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8,
        n_train_timesteps=100, n_inference_steps=100,
        lr=1e-4, betas=[0.95, 0.999], weight_decay=1e-6, warmup_steps=500,
        batch_size=256, iterations=590000,
    ),
    'dpr': dict(
        h_o=2, h_p=4, h_a=4,
        width=1728, n_blocks=3, time_embed_dim=10, time_scale=100.0, time_max_period=50.0, act='mish',
        lr=4.242224462e-4, batch_size=1088, iterations=262144,
        weight_decay=0.01, grad_clip=0.5, warmup_frac=0.05,
        n_steps=100,
    ),
    'cp': dict(                                         # Consistency Policy, adapted to states
        h_o=2, h_p=16, h_a=8, act_dim=10, normalization='dp',
        step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8, dropout=0.2,
        lr=1e-4, betas=[0.95, 0.999], weight_decay=1e-6, warmup_steps=500, batch_size=256,
        iterations_teacher=150000, iterations_student=150000, lr_schedule_student=120000,
        chaining_times=[27.0, 54.0],
    ),
}
