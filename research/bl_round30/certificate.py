"""Independent directed-Decimal enclosure of the finite objective gradient.

All inputs are exact binary floats. Rational batch coefficients retain repeated
row multiplicity and each batch's denominator. Decimal arithmetic uses directed
rounding; correctly rounded Decimal.exp is padded by adjacent representables.
The bound applies to the real affine/softplus/KL/L2 objective, never fp32 actions.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Context, Decimal
from fractions import Fraction
from time import monotonic
from typing import Any

import numpy as np


class Intervals:
    """Small scalar interval arithmetic, including outward transcendental padding."""

    def __init__(self) -> None:
        self.lo = Context(prec=50, rounding=ROUND_FLOOR)
        self.hi = Context(prec=50, rounding=ROUND_CEILING)

    def point(self, value: Any) -> tuple[Decimal, Decimal]:
        x = Decimal.from_float(float(value))
        return x, x

    def rational(self, value: Fraction) -> tuple[Decimal, Decimal]:
        a, b = Decimal(value.numerator), Decimal(value.denominator)
        return self.lo.divide(a, b), self.hi.divide(a, b)

    def add(self, a: tuple, b: tuple) -> tuple:
        return self.lo.add(a[0], b[0]), self.hi.add(a[1], b[1])

    def neg(self, a: tuple) -> tuple:
        return a[1].copy_negate(), a[0].copy_negate()

    def sub(self, a: tuple, b: tuple) -> tuple:
        return self.add(a, self.neg(b))

    def mul(self, a: tuple, b: tuple) -> tuple:
        return (
            min(self.lo.multiply(x, y) for x in a for y in b),
            max(self.hi.multiply(x, y) for x in a for y in b),
        )

    def div(self, a: tuple, b: tuple) -> tuple:
        if b[0] <= 0 <= b[1]:
            raise ArithmeticError("interval denominator includes zero")
        reciprocal = (self.lo.divide(Decimal(1), b[1]), self.hi.divide(Decimal(1), b[0]))
        return self.mul(a, reciprocal)

    def exp(self, a: tuple) -> tuple:
        return (self.lo.next_minus(self.lo.exp(a[0])), self.hi.next_plus(self.hi.exp(a[1])))

    def sigmoid(self, a: tuple) -> tuple:
        one = self.point(1)

        def scalar(x: Decimal) -> tuple:
            if x >= 0:
                return self.div(one, self.add(one, self.exp((x.copy_negate(), x.copy_negate()))))
            e = self.exp((x, x))
            return self.div(e, self.add(one, e))

        return scalar(a[0])[0], scalar(a[1])[1]

    def softmax(self, q: list[tuple]) -> list[tuple]:
        shift = max(x[1] for x in q)
        values = [self.exp(self.sub(x, (shift, shift))) for x in q]
        total = self.point(0)
        for value in values:
            total = self.add(total, value)
        return [self.div(value, total) for value in values]


def gradient_certificate(
    parameters: Any,
    cache: dict,
    plan: Any,
    recipe: Any,
    *,
    successor: bool = False,
    max_seconds: float = 600,
) -> dict:
    """Enclose every gradient coordinate and the strong-convexity bound once.

    A finite 600-second certificate check is inside the stage envelope. Its
    expiration is INCOMPLETE and never permits another solver run. Using 1e-5
    as target is conservative for 1e-5*max(1,F(0)), even if F(0) rounds upward.
    """
    start = monotonic()

    def deadline() -> None:
        if monotonic() - start >= max_seconds:
            raise TimeoutError("INCOMPLETE: interval certificate wall cap")

    def array(key: str) -> np.ndarray:
        return cache[key].detach().cpu().numpy()

    x, base, mass = array("features"), array("q"), array("mass")
    mask = array("action_mask").copy()
    mask[~mask.any(1), :3] = True
    legal, exact, died, ordinary = (array(k) for k in ("legal", "status", "died", "ordinary"))
    known = mask & legal & exact
    survive, die = known & ~died, known & died
    complete = ((~(mask & legal)) | (exact & np.isfinite(mass))).all(1)
    bare = np.where(mask, base, -np.inf).argmax(1)
    rescue = ~ordinary & complete & die[np.arange(len(x)), bare]
    pairs = survive[:, :, None] & die[:, None, :]
    growth = (
        survive[:, :, None]
        & survive[:, None, :]
        & ordinary[:, None, None]
        & (mass[:, :, None] > mass[:, None, :])
    )
    extra = (
        survive[:, :, None]
        & survive[:, None, :]
        & rescue[:, None, None]
        & (mass[:, :, None] > mass[:, None, :])
    )
    masks = (pairs, growth, ordinary[:, None], extra)
    row_counts = [m.reshape(len(x), -1).sum(1) for m in masks]
    coefficients = [[Fraction(0) for _ in x] for _ in masks]
    batches = plan.indices.cpu().numpy()
    for indices in batches:
        deadline()
        occurrences = np.bincount(indices, minlength=len(x))
        for k, count in enumerate(row_counts):
            denominator = int(count[indices].sum()) * len(batches)
            if denominator:
                for i in np.flatnonzero(occurrences):
                    coefficients[k][i] += Fraction(int(occurrences[i]), denominator)
    interval = Intervals()
    zero = interval.point(0)
    theta = parameters.detach().cpu().double().numpy()
    weights = [[interval.point(v) for v in theta[a * 256 : (a + 1) * 256]] for a in range(6)]
    bias = [interval.point(v) for v in theta[1536:]]
    gradient = [zero for _ in range(1542)]
    for i in range(len(x)):
        deadline()
        features = [interval.point(v) for v in x[i]]
        q = []
        for a in range(6):
            value = interval.add(interval.point(base[i, a]), bias[a])
            for f in range(256):
                value = interval.add(value, interval.mul(weights[a][f], features[f]))
            q.append(value)
        dq = [zero for _ in range(6)]
        for k, pair_mask in ((0, pairs[i]), (1, growth[i]), (3, extra[i])):
            if k == 3 and not successor:
                continue
            coefficient = interval.rational(coefficients[k][i])
            multiplier = recipe.preference_weight if k == 0 else 1.0
            if k == 1:
                multiplier = recipe.ordinary_growth_weight
            coefficient = interval.mul(coefficient, interval.point(multiplier))
            for good, poor in zip(*np.where(pair_mask)):
                contribution = interval.mul(
                    coefficient, interval.sigmoid(interval.sub(q[poor], q[good]))
                )
                if k != 0:
                    gap = interval.sub(interval.point(mass[i, good]), interval.point(mass[i, poor]))
                    cap = interval.point(recipe.max_mass_gap)
                    gap = min(gap[0], cap[0]), min(gap[1], cap[1])
                    contribution = interval.mul(contribution, interval.div(gap, cap))
                dq[poor] = interval.add(dq[poor], contribution)
                dq[good] = interval.sub(dq[good], contribution)
        if ordinary[i]:
            predicted = interval.softmax(q)
            target = interval.softmax([interval.point(v) for v in base[i]])
            coefficient = interval.mul(
                interval.rational(coefficients[2][i]), interval.point(recipe.anchor_weight)
            )
            for a in range(6):
                dq[a] = interval.add(
                    dq[a], interval.mul(coefficient, interval.sub(predicted[a], target[a]))
                )
        for a in range(6):
            gradient[1536 + a] = interval.add(gradient[1536 + a], dq[a])
            for f in range(256):
                j = a * 256 + f
                gradient[j] = interval.add(gradient[j], interval.mul(dq[a], features[f]))
    for j, parameter in enumerate(theta):
        coefficient = interval.div(
            interval.point(2 * recipe.residual_weight), interval.point(1536 if j < 1536 else 6)
        )
        gradient[j] = interval.add(
            gradient[j], interval.mul(coefficient, interval.point(parameter))
        )
    norm_sq = Decimal(0)
    for lower, upper in gradient:
        magnitude = max(lower.copy_abs(), upper.copy_abs())
        norm_sq = interval.hi.add(norm_sq, interval.hi.multiply(magnitude, magnitude))
    mu = interval.lo.divide(Decimal.from_float(2 * recipe.residual_weight), Decimal(1536))
    bound = interval.hi.divide(norm_sq, interval.lo.multiply(Decimal(2), mu))
    # Decimal->float conversion padded upward before exposing a JSON number.
    upper_bound = float(np.nextafter(float(bound), np.inf))
    deadline()
    return {
        "method": "directed-decimal-gradient/v1",
        "precision": 50,
        "gradient_norm_sq_upper": float(np.nextafter(float(norm_sq), np.inf)),
        "strong_convexity_mu_lower": float(np.nextafter(float(mu), -np.inf)),
        "certificate_bound": upper_bound,
        "target": 1e-5,
        "certified": bound <= Decimal("0.00001"),
        "elapsed_seconds": monotonic() - start,
        "numerical_error_included": True,
    }
