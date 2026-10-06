"""Speculative decoding (brief 20A2): the closed forms, and the acceptance draws the simulator uses.

A draft model proposes ``gamma`` tokens; the target model checks all of them in one forward pass (``CostModel
.step_spec``) and keeps the longest accepted run plus one token of its own (the correction after a rejection, or a
bonus token when every draft is accepted). With an acceptance rate ``alpha`` per drafted token, independent across
positions (Leviathan, Kalman and Matias, arXiv:2211.17192, section 3.1), the tokens one verify pass yields are

    E[tokens] = (1 - alpha^(gamma + 1)) / (1 - alpha)                         (their equation 1)

and, if one draft step costs ``c`` target steps, the expected wall-time improvement is

    (1 - alpha^(gamma + 1)) / ((1 - alpha) (gamma c + 1))                     (their Theorem 3.8)

while the total arithmetic grows by (1 - alpha)(gamma c^ + gamma + 1) / (1 - alpha^(gamma + 1)) (Theorem 3.11): the
rejected positions are wasted work, which is why speculation pays when decode is memory-bound and costs when it is
compute-bound.

The simulator draws each row's accepted run from a per-request 32-bit generator (mulberry32, the one the JavaScript
port's workload uses), so Python and JavaScript draw the same numbers: a drafted token is accepted while u < alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import ipow

MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class Speculative:
    """``draft``: a MODELS key (a smaller model with the target's vocabulary) or ``"mtp"``, one extra layer of the
    target that shares its embedding and LM head (DeepSeek-V3's multi-token prediction module, arXiv:2412.19437;
    EAGLE's draft head is similar, arXiv:2401.15077). ``gamma`` drafted tokens per verify pass, acceptance rate
    ``alpha`` per drafted token (a parameter: it depends on the draft, the target and the text), ``seed`` for the
    acceptance draws."""

    draft: str = "mtp"
    gamma: int = 3
    alpha: float = 0.7
    seed: int = 0


def expected_tokens(alpha: float, gamma: int) -> float:
    """Tokens per verify pass: (1 - alpha^(gamma+1)) / (1 - alpha); gamma + 1 when alpha = 1."""
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - ipow(alpha, gamma + 1)) / (1.0 - alpha)


def expected_speedup(alpha: float, gamma: int, c: float) -> float:
    """Leviathan et al. Theorem 3.8: wall-time improvement when a draft step costs c target steps."""
    return expected_tokens(alpha, gamma) / (gamma * c + 1.0)


def expected_operations(alpha: float, gamma: int, c_hat: float) -> float:
    """Leviathan et al. Theorem 3.11: the factor by which total arithmetic grows, when one draft token costs c_hat
    of a target token's operations: (1 - alpha)(gamma c_hat + gamma + 1) / (1 - alpha^(gamma+1))."""
    return (gamma * c_hat + gamma + 1.0) / expected_tokens(alpha, gamma)


def _imul(a: int, b: int) -> int:
    return (a * b) & MASK


def seed_state(seed: int, rid: int) -> int:
    """The generator state of request ``rid`` (both languages: below 2^53 before the modulo)."""
    return (seed + rid * 2654435761) % 4294967296


def mulberry32(a: int) -> tuple[int, float]:
    """One step of mulberry32: (new state, uniform in [0, 1)). Bit-for-bit the JavaScript function."""
    a = (a + 0x6D2B79F5) & MASK
    t = _imul(a ^ (a >> 15), 1 | a)
    t = ((t + _imul(t ^ (t >> 7), 61 | t)) & MASK) ^ t
    return a, ((t ^ (t >> 14)) & MASK) / 4294967296


def draw_accepted(state: int, alpha: float, gamma: int) -> tuple[int, int]:
    """(new state, drafted tokens accepted in one verify pass): draws until a rejection or gamma acceptances."""
    k = 0
    while k < gamma:
        state, u = mulberry32(state)
        if u >= alpha:
            break
        k += 1
    return state, k
