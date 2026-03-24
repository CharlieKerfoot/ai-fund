"""Daily HTML report generator using Jinja2 templating."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from jinja2 import Template

logger = logging.getLogger(__name__)

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Fund Daily Report - {{ report_date }}</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
    h1 { color: #333; }
    h2 { color: #555; border-bottom: 1px solid #ccc; padding-bottom: 4px; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
    th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
    th { background: #eee; }
    .badge { padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    .degraded { background: #f8d7da; color: #721c24; }
    .normal { background: #d4edda; color: #155724; }
    .regime { background: #d1ecf1; color: #0c5460; }
  </style>
</head>
<body>

<h1>AI Fund Daily Report</h1>

<section>
  <h2>Header</h2>
  <p><strong>Date:</strong> {{ report_date }}</p>
  <p><strong>Regime:</strong> <span class="badge regime">{{ regime }}</span></p>
  <p><strong>Mode:</strong>
    {% if degraded_mode %}
      <span class="badge degraded">DEGRADED</span>
    {% else %}
      <span class="badge normal">Normal</span>
    {% endif %}
  </p>
</section>

<section>
  <h2>Portfolio Summary</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Portfolio Value</td><td>${{ "%.2f"|format(portfolio_value) }}</td></tr>
    <tr><td>Cash</td><td>${{ "%.2f"|format(cash) }}</td></tr>
    <tr><td>Gross P&amp;L</td><td>${{ "%.2f"|format(gross_pnl) }}</td></tr>
    <tr><td>Net P&amp;L</td><td>${{ "%.2f"|format(net_pnl) }}</td></tr>
    <tr><td>Daily Return</td><td>{{ "%.4f"|format(daily_return) }}%</td></tr>
  </table>
</section>

<section>
  <h2>Positions</h2>
  {% if positions %}
  <table>
    <tr>
      <th>Symbol</th>
      <th>Quantity</th>
      <th>Market Value</th>
      <th>Weight</th>
    </tr>
    {% for pos in positions %}
    <tr>
      <td>{{ pos.symbol }}</td>
      <td>{{ "%.2f"|format(pos.quantity) }}</td>
      <td>${{ "%.2f"|format(pos.market_value) }}</td>
      <td>{{ "%.2f"|format(pos.weight * 100) }}%</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p>No open positions.</p>
  {% endif %}
</section>

<section>
  <h2>Signal Performance</h2>
  {% if signal_summary %}
  <table>
    <tr>
      <th>Signal</th>
      <th>Hit Rate</th>
      <th>Attributed P&amp;L</th>
      <th>Sharpe Contribution</th>
    </tr>
    {% for sig in signal_summary %}
    <tr>
      <td>{{ sig.signal_name }}</td>
      <td>{{ "%.2f"|format(sig.hit_rate * 100) }}%</td>
      <td>${{ "%.2f"|format(sig.attributed_pnl) }}</td>
      <td>{{ "%.4f"|format(sig.sharpe_contribution) if sig.sharpe_contribution == sig.sharpe_contribution else "N/A" }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p>No signal data available.</p>
  {% endif %}
</section>

<section>
  <h2>Risk Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td>VaR (95%, 1-day)</td><td>{{ var_95 if var_95 is not none else "N/A" }}</td></tr>
    <tr><td>Worst Stress Test</td><td>{{ worst_stress }}</td></tr>
  </table>
</section>

<section>
  <h2>Equity Curve (Last 30 Days)</h2>
  {% if equity_curve %}
  <table>
    <tr><th>Date</th><th>Portfolio Value</th></tr>
    {% for row in equity_curve %}
    <tr>
      <td>{{ row.date }}</td>
      <td>${{ "%.2f"|format(row.value) }}</td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
  <p>No equity curve data available.</p>
  {% endif %}
</section>

</body>
</html>
"""


class DailyReporter:
    def __init__(self, conn, reports_dir: str) -> None:
        self._conn = conn
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        self._template = Template(_TEMPLATE)

    def generate(
        self,
        report_date: date,
        regime: str = "unknown",
        degraded_mode: bool = False,
    ) -> str:
        """Generate HTML report and save to reports/{date}.html.

        Returns the file path.
        """
        # Portfolio summary
        pnl_row = self._conn.execute(
            "SELECT gross_pnl, net_pnl, transaction_costs, portfolio_value, cash "
            "FROM daily_pnl WHERE date = ?",
            [report_date],
        ).fetchone()

        if pnl_row:
            gross_pnl, net_pnl, transaction_costs, portfolio_value, cash = pnl_row
        else:
            gross_pnl = net_pnl = 0.0
            portfolio_value = 0.0
            cash = 0.0

        # Daily return
        yesterday = report_date - timedelta(days=1)
        prev_row = self._conn.execute(
            "SELECT portfolio_value FROM daily_pnl WHERE date = ?",
            [yesterday],
        ).fetchone()
        if prev_row and prev_row[0] and prev_row[0] > 0:
            daily_return = (portfolio_value - prev_row[0]) / prev_row[0] * 100.0
        else:
            daily_return = 0.0

        # Positions
        positions_df = self._conn.execute(
            """
            SELECT symbol, SUM(quantity) AS quantity
            FROM fills
            WHERE date <= ?
            GROUP BY symbol
            HAVING SUM(quantity) != 0
            """,
            [report_date],
        ).df()

        positions = []
        total_pos_value = 0.0
        if not positions_df.empty:
            # Get latest prices from fills
            for _, row in positions_df.iterrows():
                sym = row["symbol"]
                qty = float(row["quantity"])
                price_row = self._conn.execute(
                    "SELECT fill_price FROM fills WHERE symbol = ? AND date <= ? "
                    "ORDER BY date DESC, filled_at DESC LIMIT 1",
                    [sym, report_date],
                ).fetchone()
                price = float(price_row[0]) if price_row else 0.0
                market_value = qty * price
                total_pos_value += market_value
                positions.append({
                    "symbol": sym,
                    "quantity": qty,
                    "market_value": market_value,
                    "weight": 0.0,  # computed below
                })

            if portfolio_value > 0:
                for pos in positions:
                    pos["weight"] = pos["market_value"] / portfolio_value

        # Signal summary
        signal_summary_df = self._conn.execute(
            """
            SELECT DISTINCT signal_id FROM signals
            WHERE date = ?
            """,
            [report_date],
        ).df()

        signal_summary = []
        if not signal_summary_df.empty:
            for sig_id in signal_summary_df["signal_id"].tolist():
                signal_summary.append({
                    "signal_name": sig_id,
                    "hit_rate": 0.5,
                    "attributed_pnl": 0.0,
                    "sharpe_contribution": float("nan"),
                })

        # Risk metrics (placeholders if no VaR computed)
        var_95 = "N/A"
        worst_stress = "N/A"

        # Equity curve: last 30 days
        start_30 = report_date - timedelta(days=30)
        equity_df = self._conn.execute(
            """
            SELECT date, portfolio_value
            FROM daily_pnl
            WHERE date >= ? AND date <= ?
            ORDER BY date
            """,
            [start_30, report_date],
        ).df()

        equity_curve = []
        if not equity_df.empty:
            for _, row in equity_df.iterrows():
                equity_curve.append({
                    "date": str(row["date"]),
                    "value": float(row["portfolio_value"]),
                })

        html = self._template.render(
            report_date=str(report_date),
            regime=regime,
            degraded_mode=degraded_mode,
            portfolio_value=portfolio_value or 0.0,
            cash=cash or 0.0,
            gross_pnl=gross_pnl or 0.0,
            net_pnl=net_pnl or 0.0,
            daily_return=daily_return,
            positions=positions,
            signal_summary=signal_summary,
            var_95=var_95,
            worst_stress=worst_stress,
            equity_curve=equity_curve,
        )

        output_path = self._reports_dir / f"{report_date}.html"
        output_path.write_text(html, encoding="utf-8")
        logger.info("Daily report written to %s", output_path)

        return str(output_path)
