# Author: Mohammad Al-Jarrah
"""
The policies of the paper, with one interface:

    policy = build_policy(name, cfg, dim_s, dim_a)     # an nn.Module
    policy.fit(S, A)                                   # normalized training pairs (torch tensors)
    policy.sample(s, z)                                # action chunks for states s and reference
                                                       # samples z (ignored by Gaussian-reference policies)

Proposed policies (reference P_A, the empirical distribution of demonstrated chunks):
    otp  optimal transport policy        fmp  flow matching policy     sip  stochastic interpolant policy
Gaussian-reference policies:
    krp  Knothe-Rosenblatt policy        dp   Diffusion Policy         dpr  DP objective on the residual MLP
    cp   Consistency Policy (1-step; 3-step with chaining)
"""
from .otp import OTP
from .fmp import FMP
from .sip import SIP
from .krp import KRP
from .dp import DP
from .dpr import DPR
from .cp import CP
from .agent import Agent, save_checkpoint, load_checkpoint

POLICIES = {'otp': OTP, 'fmp': FMP, 'sip': SIP, 'krp': KRP, 'dp': DP, 'dpr': DPR, 'cp': CP}
NAMES = {'otp': 'OTP', 'fmp': 'FMP', 'sip': 'SIP', 'krp': 'KRP', 'dp': 'DP', 'dpr': 'DPR', 'cp': 'CP'}


def build_policy(name, cfg, dim_s, dim_a):
    """An untrained policy `name` with settings cfg, state dimension dim_s and chunk dimension dim_a."""
    return POLICIES[name](cfg, dim_s, dim_a)
