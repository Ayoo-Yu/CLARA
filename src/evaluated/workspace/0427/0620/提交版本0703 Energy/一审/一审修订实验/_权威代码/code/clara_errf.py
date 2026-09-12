from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrfTheta:
    pi_plus: float = 3.56
    pi_minus: float = 3.56
    kappa_plus: float = 20.0
    kappa_minus: float = 20.0
    theta_id: str = "Guangxi_AGC_price_anchor"


@dataclass(frozen=True)
class ErrfResult:
    reserve_up: float
    reserve_down: float
    miss_upper: float
    miss_lower: float
    total: float
    covered: bool


def compute_event_errf(
    *,
    target: float,
    schedule: float,
    lower: float,
    upper: float,
    nominal_cadence_minutes: float,
    theta: ErrfTheta,
    capacity_scale_mw: float = 1.0,
) -> ErrfResult:
    values = [target, schedule, lower, upper, nominal_cadence_minutes, capacity_scale_mw]
    if any(value != value for value in values):
        raise ValueError("ERRF输入含非有限值")
    if lower > upper:
        raise ValueError("ERRF区间下界大于上界")
    if nominal_cadence_minutes <= 0.0 or capacity_scale_mw <= 0.0:
        raise ValueError("ERRF时间步长和容量尺度必须为正")
    scale = float(nominal_cadence_minutes) / 60.0 * float(capacity_scale_mw)
    reserve_up = scale * max(float(upper) - float(schedule), 0.0) * float(theta.pi_plus)
    reserve_down = scale * max(float(schedule) - float(lower), 0.0) * float(theta.pi_minus)
    miss_upper = scale * max(float(target) - float(upper), 0.0) * float(theta.kappa_plus)
    miss_lower = scale * max(float(lower) - float(target), 0.0) * float(theta.kappa_minus)
    total = reserve_up + reserve_down + miss_upper + miss_lower
    return ErrfResult(
        reserve_up=reserve_up,
        reserve_down=reserve_down,
        miss_upper=miss_upper,
        miss_lower=miss_lower,
        total=total,
        covered=float(lower) <= float(target) <= float(upper),
    )
