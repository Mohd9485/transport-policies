# Author: Mohammad Al-Jarrah
"""
SMAC3 search spaces of the tuned policies (OTP, FMP, SIP, KRP, DPR).

Every space takes its DEFAULT configuration from configs/<task>.py, i.e. the settings reported in
the paper, so that a search starts from (and first evaluates) the paper's configuration. Each
entry below maps one searched hyperparameter onto the configuration:

    (configuration key, kind, choices or bounds, scale)

kind 'cat' : categorical over `choices`;
kind 'int' : integer in [low, high], multiplied by `scale` (e.g. widths in multiples of 64);
kind 'log' : float in [low, high] on a logarithmic scale;
kind 'float': float in [low, high].
For OTP, the keys 'width' and 'lr' are pairs [critic, map]; they are searched as two entries
with suffixes '_0' and '_1'. The executed horizon h_a always equals the predicted horizon h_p.
DP and CP are not tuned: they use the published settings of their authors.
"""
import copy

H_P = ('h_p', 'cat', [2, 4, 8, 16], 1)

SPACES = {
    'otp': [('lr_0', 'log', (1e-5, 2e-3), 1), ('lr_1', 'log', (1e-5, 2e-3), 1),
            ('width_0', 'int', (1, 24), 64), ('width_1', 'int', (1, 24), 64),
            ('n_blocks_0', 'cat', [1, 2, 3, 4], 1), ('n_blocks_1', 'cat', [1, 2, 3, 4], 1),
            ('batch_size', 'int', (6, 20), 64), ('k_in', 'int', (4, 6), 5),
            ('iterations', 'cat', [8, 16, 32], 1024),
            ('weight_decay', 'cat', [0.0, 1e-3, 1e-2, 1e-1], 1),
            ('grad_clip', 'cat', [0.02, 0.05, 0.1, 0.25, 0.5, 1.0], 1),
            ('warmup_frac', 'cat', [0.0, 0.02, 0.05, 0.1], 1), H_P],
    'fmp': [('lr', 'log', (1e-5, 1e-2), 1), ('width', 'int', (2, 28), 64),
            ('n_blocks', 'cat', [1, 2, 3, 4, 5, 6], 1), ('batch_size', 'int', (2, 20), 64),
            ('iterations', 'cat', [8, 16, 32, 64, 128, 256], 1024),
            ('time_embed_dim', 'int', (1, 4), 10), ('time_scale', 'int', (1, 5), 50),
            ('weight_decay', 'cat', [0.0, 1e-3, 1e-2, 1e-1, 2e-1], 1),
            ('grad_clip', 'cat', [0.1, 0.25, 0.5, 1.0, 2.0], 1),
            ('warmup_frac', 'cat', [0.0, 0.02, 0.05], 1), ('act', 'cat', ['mish', 'relu'], 1), H_P],
    'krp': [('lr', 'log', (1e-4, 1e-2), 1), ('width', 'int', (1, 6), 64), ('depth', 'cat', [2, 3, 4, 5], 1),
            ('batch_size', 'int', (2, 10), 64), ('iterations', 'cat', [8, 16, 32, 48, 64], 1024),
            ('quad_points', 'cat', [8, 16, 24], 1), ('act', 'cat', ['tanh', 'silu', 'mish'], 1),
            ('residual', 'cat', [False, True], 1),
            ('weight_decay', 'cat', [0.0, 1e-3, 1e-2, 1e-1], 1),
            ('grad_clip', 'cat', [0.25, 1.0, 5.0, 10.0], 1),
            ('warmup_frac', 'cat', [0.0, 0.02, 0.05, 0.1], 1), H_P],
}
SPACES['sip'] = SPACES['fmp'] + [('noise_level', 'float', (0.01, 0.2), 1)]
SPACES['dpr'] = SPACES['fmp']
EMA = ('ema_decay', 'cat', [0.0, 0.999, 0.9999], 1)        # searched on Square (not for DPR)


def _value(cfg, key):
    """The configuration value of a searched key ('lr_0' is cfg['lr'][0])."""
    if key[-2:] in ('_0', '_1') and isinstance(cfg.get(key[:-2]), list):
        return cfg[key[:-2]][int(key[-1])]
    return cfg[key]


def searched(task, policy):
    """The searched entries of `policy` on `task`."""
    return SPACES[policy] + ([EMA] if task == 'square' and policy != 'dpr' else [])


def build_space(task, policy, cfg):
    """The ConfigurationSpace of `policy` on `task`, whose default is the configuration `cfg`."""
    from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer
    entries = searched(task, policy)
    cs = ConfigurationSpace()
    for key, kind, rng, scale in entries:
        default = _value(cfg, key)
        if kind == 'cat':
            d = default if scale == 1 else int(default // scale)
            assert d in rng, (policy, key, default)            # the paper's value must be a choice
            cs.add(Categorical(key, list(rng), default=d))
        elif kind == 'int':
            assert default % scale == 0 and rng[0] <= default // scale <= rng[1], (policy, key, default)
            cs.add(Integer(key, rng, default=default // scale))
        else:
            assert rng[0] <= default <= rng[1], (policy, key, default)
            cs.add(Float(key, rng, log=(kind == 'log'), default=default))
    return cs


def to_config(task, policy, base, sample):
    """The configuration of one SMAC sample, on top of the configuration `base`."""
    cfg = copy.deepcopy(base)
    for key, kind, _, scale in searched(task, policy):
        v = sample[key]
        v = v * scale if kind in ('int', 'cat') and scale != 1 else v
        v = v.item() if hasattr(v, 'item') else v
        if key[-2:] in ('_0', '_1') and isinstance(cfg.get(key[:-2]), list):
            cfg[key[:-2]][int(key[-1])] = v
        else:
            cfg[key] = v
    cfg['h_a'] = cfg['h_p']
    return cfg
