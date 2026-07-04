r"""P&L decomposition: spread capture vs inventory P&L vs funding.

Identity (continuous form, integrated over [0, T]):

    Delta PnL = Sum_{fills} (mid_t - fill_price) * (-signed_size)
                  ^ spread capture
              + integral of q_{t-} d mid_t
                  ^ inventory P&L
              + Sum_{funding} (-q_t * mid_t * f_t * dt / T_f)
                  ^ funding P&L (perpetual funding transfers, settle in cash)

In discrete form, with mid sampled at every event,

    spread_capture += (fill_price - mid_at_fill) * size  for sell fills
                     (mid_at_fill - fill_price) * size  for buy fills
    inventory_pnl  += q_{t-} * (mid_t - mid_{t-1})
    funding_pnl    += -q_t * mid_t * f_t * dt / T_f

Total PnL:  spread_capture + inventory_pnl + funding_pnl == cash_t + q_t * mid_t

The class enforces the identity numerically as a class invariant.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PnLDecomposition:
    spread_capture: float
    inventory_pnl: float
    total: float
    funding_pnl: float = 0.0
    tol: float = 1e-6

    def __post_init__(self) -> None:
        residual = self.spread_capture + self.inventory_pnl + self.funding_pnl - self.total
        if abs(residual) > self.tol:
            raise AssertionError(
                f"PnL decomposition identity violated: "
                f"spread + inventory + funding − total = {residual!r} (tol={self.tol})"
            )
