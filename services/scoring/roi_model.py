# services/scoring/roi_model.py
import numpy as np
from dataclasses import dataclass

@dataclass
class ROIResult:
    conservative: float
    base: float
    optimistic: float
    dev_cost: float
    net: float
    payback_months: float
    derivation: dict

def monte_carlo_roi(arr: float, reach_pct: float,
                    effort_weeks: float, eng: int = 3,
                    eng_cost_wk: float = 8000,
                    n: int = 5000) -> ROIResult:
    rng = np.random.default_rng(42)
    reach   = rng.normal(reach_pct, reach_pct * 0.15, n).clip(0, 1)
    churn_p = rng.uniform(0.4, 0.85, n)
    effort  = rng.normal(effort_weeks, effort_weeks * 0.2, n).clip(1, 52)
    revenue = arr * reach * churn_p
    cost    = effort * eng * eng_cost_wk
    net     = revenue - cost
    base_cost = effort_weeks * eng * eng_cost_wk
    base_net  = float(np.percentile(net, 50))
    return ROIResult(
        conservative=float(np.percentile(net, 10)),
        base=base_net,
        optimistic=float(np.percentile(net, 90)),
        dev_cost=base_cost,
        net=base_net,
        payback_months=round(base_cost / (base_net / 12), 1) if base_net > 0 else 99,
        derivation={
            "arr_base": f"${arr:,.0f}",
            "reach": f"{reach_pct:.1%} of users",
            "effort": f"{effort_weeks}wks × {eng} engineers × ${eng_cost_wk:,}/wk",
            "simulations": n,
        }
    )   