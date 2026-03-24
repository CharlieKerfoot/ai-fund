"""Backtest report generation utilities."""

from __future__ import annotations

from research.backtest.engine import BacktestResult
from research.backtest.overfit import OverfitReport
from research.backtest.validation import WalkForwardResult


def generate_backtest_report(
    result: BacktestResult,
    overfit: OverfitReport | None = None,
    wf: WalkForwardResult | None = None,
) -> str:
    """Generate a plain-text report summarizing backtest results."""
    lines: list[str] = []

    lines.append("=" * 60)
    lines.append(f"BACKTEST REPORT: {result.signal_name} v{result.signal_version}")
    lines.append("=" * 60)
    lines.append(f"Period     : {result.start_date} -> {result.end_date}")
    lines.append("")

    lines.append("--- Performance Metrics ---")
    lines.append(f"Annual Return      : {result.annual_return * 100:.2f}%")
    lines.append(f"Annual Volatility  : {result.annual_volatility * 100:.2f}%")
    lines.append(f"Sharpe Ratio       : {result.sharpe:.4f}")
    lines.append(f"Sortino Ratio      : {result.sortino:.4f}")
    lines.append(f"Max Drawdown       : {result.max_drawdown * 100:.2f}%")
    lines.append(f"Calmar Ratio       : {result.calmar:.4f}")
    lines.append(f"Avg Daily Turnover : {result.turnover * 100:.2f}%")
    lines.append(f"Total TC Cost      : ${result.transaction_costs_total:,.0f}")
    lines.append(f"Survivorship Bias  : {'YES (caution)' if result.survivorship_bias_flag else 'No'}")
    lines.append("")

    if overfit is not None:
        lines.append("--- Overfit Analysis ---")
        lines.append(f"Trials Tested      : {overfit.n_trials}")
        lines.append(f"Deflated Sharpe    : {overfit.deflated_sharpe:.4f}  ({'significant' if overfit.dsr_significant else 'NOT significant'})")
        lines.append(f"Bonferroni p-val   : {overfit.bonferroni_correction:.4f}")
        lines.append(f"BH Significant     : {overfit.bh_significant}")
        lines.append(f"Min Backtest Years : {overfit.min_backtest_years:.2f}")
        lines.append(f"Actual Years       : {overfit.actual_backtest_years:.2f}  ({'OK' if overfit.sufficient_history else 'INSUFFICIENT'})")
        lines.append(f"Recommendation     : {overfit.recommendation}")
        lines.append("")

    if wf is not None:
        lines.append("--- Walk-Forward Validation ---")
        lines.append(f"IS Sharpe          : {wf.is_sharpe:.4f}")
        lines.append(f"OOS Sharpe (mean)  : {wf.oos_sharpe:.4f}")
        lines.append(f"OOS Sharpe (std)   : {wf.std_oos_sharpe:.4f}")
        lines.append(f"Degradation Ratio  : {wf.degradation_ratio:.4f}  ({'healthy' if wf.degradation_ratio >= 0.5 else 'degraded'})")
        lines.append("")
        lines.append("  Fold | Train End   | Test Period                   | OOS Sharpe")
        lines.append("  " + "-" * 65)
        for fold in wf.folds:
            skipped = fold.get("skipped", False)
            sharpe_str = f"{fold['sharpe']:.4f}" if not skipped else "SKIPPED"
            lines.append(
                f"  {fold['fold']:>4d} | {fold['train_end']} | {fold['test_start']} -> {fold['test_end']} | {sharpe_str}"
            )
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def equity_curve_to_html(result: BacktestResult) -> str:
    """Generate an HTML table with the equity curve and key metrics."""
    # Metrics summary
    metrics_rows = [
        ("Annual Return", f"{result.annual_return * 100:.2f}%"),
        ("Annual Volatility", f"{result.annual_volatility * 100:.2f}%"),
        ("Sharpe Ratio", f"{result.sharpe:.4f}"),
        ("Sortino Ratio", f"{result.sortino:.4f}"),
        ("Max Drawdown", f"{result.max_drawdown * 100:.2f}%"),
        ("Calmar Ratio", f"{result.calmar:.4f}"),
        ("Avg Daily Turnover", f"{result.turnover * 100:.2f}%"),
        ("Total TC Cost", f"${result.transaction_costs_total:,.0f}"),
    ]

    metrics_html_rows = "\n".join(
        f"    <tr><td>{k}</td><td>{v}</td></tr>" for k, v in metrics_rows
    )

    # Equity curve table (sample every ~20 rows to keep HTML compact)
    ec = result.equity_curve
    step = max(1, len(ec) // 50)
    sampled = ec.iloc[::step]

    eq_rows = "\n".join(
        f"    <tr><td>{ts.date() if hasattr(ts, 'date') else ts}</td><td>${val:,.0f}</td></tr>"
        for ts, val in sampled.items()
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
  <title>Backtest Report: {result.signal_name}</title>
  <style>
    body {{ font-family: monospace; padding: 1em; }}
    h2 {{ margin-bottom: 0.5em; }}
    table {{ border-collapse: collapse; margin-bottom: 2em; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
    th {{ background: #f0f0f0; }}
  </style>
</head>
<body>
  <h2>Backtest Report: {result.signal_name} v{result.signal_version}</h2>
  <p>Period: {result.start_date} &rarr; {result.end_date}</p>

  <h3>Performance Metrics</h3>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
{metrics_html_rows}
  </table>

  <h3>Equity Curve (sampled)</h3>
  <table>
    <tr><th>Date</th><th>Portfolio Value</th></tr>
{eq_rows}
  </table>
</body>
</html>"""
    return html
