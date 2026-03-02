"""Monte Carlo simulation for prop firm pass probability.

Takes a list of trade returns, reshuffles them thousands of times,
and calculates the probability of passing a prop firm challenge at
different drawdown levels.
"""

import numpy as np
import pandas as pd


class MonteCarloSimulator:

    def __init__(self, config: dict):
        self.n_simulations = config["monte_carlo"]["n_simulations"]  # 10000
        self.dd_levels = config["monte_carlo"]["dd_levels"]          # [0.06, 0.10]

    def simulate(self, trade_pnls: list[float], config: dict,
                  trades_per_day: float = 2.0) -> dict:
        """Run Monte Carlo simulation.

        Args:
            trade_pnls: List of actual trade P&L values from backtest
            config: Config with prop firm rules
            trades_per_day: Average trades per day to estimate challenge duration

        Returns:
            dict with pass probabilities, drawdown distributions, etc.
        """
        if not trade_pnls:
            return self._empty_result()

        pnls = np.array(trade_pnls)
        initial_equity = config["backtest"]["initial_equity"]

        prop = config.get("prop_firm", {})
        profit_target = prop.get("profit_target_pct", 0.08)
        challenge_days = prop.get("challenge_days", 30)
        max_daily_loss = prop.get("max_daily_loss_pct", 0.05)

        # Estimate max trades in a challenge
        max_trades = int(challenge_days * trades_per_day)
        n_trades = min(len(pnls), max_trades)

        print(f"Monte Carlo: {self.n_simulations} simulations, {n_trades} trades each")
        print(f"Profit target: {profit_target:.0%}, Challenge: {challenge_days} days")

        results = {dd: {"passed": 0, "max_dds": [], "final_equities": [], "days_to_target": []}
                   for dd in self.dd_levels}

        rng = np.random.default_rng(42)  # Reproducible

        for sim in range(self.n_simulations):
            # Shuffle trade sequence
            shuffled = rng.choice(pnls, size=n_trades, replace=True)

            # Build equity curve
            equity = initial_equity
            peak = initial_equity
            max_dd = 0
            daily_pnl = 0
            daily_trade_count = 0
            hit_target = False
            target_trade = -1

            for t_idx, pnl in enumerate(shuffled):
                equity += pnl
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

                daily_pnl += pnl
                daily_trade_count += 1

                # Check profit target
                if not hit_target and (equity - initial_equity) / initial_equity >= profit_target:
                    hit_target = True
                    target_trade = t_idx

                # Reset daily tracking every N trades (approximate)
                if daily_trade_count >= trades_per_day:
                    daily_pnl = 0
                    daily_trade_count = 0

            final_equity = equity

            # Check pass/fail for each DD level
            for dd_level in self.dd_levels:
                results[dd_level]["max_dds"].append(max_dd)
                results[dd_level]["final_equities"].append(final_equity)

                if hit_target and max_dd <= dd_level:
                    results[dd_level]["passed"] += 1
                    results[dd_level]["days_to_target"].append(target_trade / trades_per_day)

        # Compile results
        summary = {}
        for dd_level in self.dd_levels:
            r = results[dd_level]
            pass_rate = r["passed"] / self.n_simulations
            max_dds = np.array(r["max_dds"])
            finals = np.array(r["final_equities"])
            days = np.array(r["days_to_target"]) if r["days_to_target"] else np.array([0])

            summary[f"dd_{int(dd_level*100)}pct"] = {
                "pass_probability": pass_rate,
                "avg_max_drawdown": float(max_dds.mean()),
                "median_max_drawdown": float(np.median(max_dds)),
                "p95_max_drawdown": float(np.percentile(max_dds, 95)),
                "avg_final_equity": float(finals.mean()),
                "median_final_equity": float(np.median(finals)),
                "avg_days_to_target": float(days.mean()) if len(days) > 0 else 0,
                "p25_final_equity": float(np.percentile(finals, 25)),
                "p75_final_equity": float(np.percentile(finals, 75)),
            }

            print(f"\n--- Max Drawdown {dd_level:.0%} ---")
            print(f"  Pass probability: {pass_rate:.1%}")
            print(f"  Avg max DD:       {max_dds.mean():.2%}")
            print(f"  P95 max DD:       {np.percentile(max_dds, 95):.2%}")
            print(f"  Avg final equity: ${finals.mean():,.0f}")
            if r["days_to_target"]:
                print(f"  Avg days to target: {days.mean():.1f}")

        return summary

    def _empty_result(self) -> dict:
        return {f"dd_{int(dd*100)}pct": {
            "pass_probability": 0,
            "avg_max_drawdown": 0,
            "median_max_drawdown": 0,
            "p95_max_drawdown": 0,
            "avg_final_equity": 0,
            "median_final_equity": 0,
            "avg_days_to_target": 0,
        } for dd in self.dd_levels}
