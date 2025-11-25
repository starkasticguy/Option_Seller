"""
Backtesting Analytics Module
Calculates performance metrics and generates reports
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import json
from pathlib import Path

# Import plotly for interactive charts
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Install with: pip install plotly")

from backtest_config import ANALYTICS_CONFIG, DATA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceAnalytics:
    """
    Calculate comprehensive performance metrics for backtest results
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        equity_curve: pd.DataFrame,
        initial_capital: float,
        risk_free_rate: float = None
    ):
        """
        Initialize analytics

        Args:
            trades_df: DataFrame with all trades
            equity_curve: DataFrame with equity curve
            initial_capital: Initial capital
            risk_free_rate: Risk-free rate for Sharpe calculation
        """
        self.trades_df = trades_df
        self.equity_curve = equity_curve
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate or ANALYTICS_CONFIG['risk_free_rate']

        logger.info("Performance analytics initialized")

    def calculate_all_metrics(self) -> Dict:
        """
        Calculate all performance metrics

        Returns:
            Dictionary with all metrics
        """
        logger.info("Calculating performance metrics...")

        metrics = {}

        # Basic metrics
        metrics.update(self._calculate_basic_metrics())

        # Return metrics
        metrics.update(self._calculate_return_metrics())

        # Risk metrics
        metrics.update(self._calculate_risk_metrics())

        # Trade analysis
        metrics.update(self._calculate_trade_metrics())

        # Drawdown analysis
        metrics.update(self._calculate_drawdown_metrics())

        # Win/Loss analysis
        metrics.update(self._calculate_win_loss_metrics())

        logger.info("✅ Metrics calculation completed")
        return metrics

    def _calculate_basic_metrics(self) -> Dict:
        """Calculate basic metrics"""
        if self.equity_curve.empty:
            return {}

        final_capital = self.equity_curve['equity'].iloc[-1]
        total_return = (final_capital / self.initial_capital - 1) * 100

        # Calculate duration
        start_date = self.equity_curve['timestamp'].iloc[0]
        end_date = self.equity_curve['timestamp'].iloc[-1]
        duration_days = (end_date - start_date).days

        return {
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_pnl': final_capital - self.initial_capital,
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': duration_days,
            'num_trades': len(self.trades_df),
        }

    def _calculate_return_metrics(self) -> Dict:
        """Calculate return metrics"""
        if self.equity_curve.empty:
            return {}

        # Calculate returns
        equity_series = self.equity_curve.set_index('timestamp')['equity']
        returns = equity_series.pct_change().dropna()

        # Total return
        total_return = (equity_series.iloc[-1] / equity_series.iloc[0] - 1)

        # Calculate CAGR
        duration_days = (self.equity_curve['timestamp'].iloc[-1] -
                        self.equity_curve['timestamp'].iloc[0]).days
        duration_years = duration_days / 365.0

        if duration_years > 0:
            cagr = ((equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / duration_years) - 1) * 100
        else:
            cagr = 0

        # Average daily/monthly returns
        avg_daily_return = returns.mean() * 100
        avg_monthly_return = returns.mean() * 21 * 100  # Assuming 21 trading days per month

        return {
            'total_return_pct': total_return * 100,
            'cagr': cagr,
            'avg_daily_return_pct': avg_daily_return,
            'avg_monthly_return_pct': avg_monthly_return,
        }

    def _calculate_risk_metrics(self) -> Dict:
        """Calculate risk-adjusted metrics"""
        if self.equity_curve.empty:
            return {}

        # Calculate returns
        equity_series = self.equity_curve.set_index('timestamp')['equity']
        returns = equity_series.pct_change().dropna()

        if len(returns) == 0:
            return {}

        # Volatility (annualized)
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252) * 100

        # Sharpe Ratio
        excess_returns = returns - (self.risk_free_rate / 252)  # Daily risk-free rate
        if excess_returns.std() != 0:
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Sortino Ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std != 0:
                sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252)
            else:
                sortino_ratio = 0
        else:
            sortino_ratio = 0

        # Calmar Ratio (CAGR / Max Drawdown)
        max_dd = self.equity_curve['drawdown'].min()
        if max_dd != 0:
            duration_years = ((self.equity_curve['timestamp'].iloc[-1] -
                             self.equity_curve['timestamp'].iloc[0]).days / 365.0)
            cagr = ((equity_series.iloc[-1] / equity_series.iloc[0]) ** (1 / duration_years) - 1) * 100
            calmar_ratio = cagr / abs(max_dd)
        else:
            calmar_ratio = 0

        return {
            'volatility_annual_pct': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
        }

    def _calculate_drawdown_metrics(self) -> Dict:
        """Calculate drawdown metrics"""
        if self.equity_curve.empty:
            return {}

        drawdowns = self.equity_curve['drawdown']

        max_drawdown = drawdowns.min()

        # Find max drawdown period
        in_drawdown = drawdowns < 0
        dd_periods = []
        current_period = []

        for idx, is_dd in enumerate(in_drawdown):
            if is_dd:
                current_period.append(idx)
            else:
                if current_period:
                    dd_periods.append(current_period)
                    current_period = []

        if current_period:
            dd_periods.append(current_period)

        # Calculate longest drawdown
        if dd_periods:
            longest_dd_days = max(len(period) for period in dd_periods)
            num_drawdown_periods = len(dd_periods)
        else:
            longest_dd_days = 0
            num_drawdown_periods = 0

        # Calculate recovery time for max drawdown
        max_dd_idx = drawdowns.idxmin()
        recovery_idx = None

        for idx in range(max_dd_idx + 1, len(drawdowns)):
            if drawdowns.iloc[idx] >= 0:
                recovery_idx = idx
                break

        if recovery_idx:
            recovery_days = recovery_idx - max_dd_idx
        else:
            recovery_days = None  # Not recovered yet

        return {
            'max_drawdown_pct': max_drawdown,
            'longest_dd_days': longest_dd_days,
            'num_drawdown_periods': num_drawdown_periods,
            'recovery_days': recovery_days,
        }

    def _calculate_trade_metrics(self) -> Dict:
        """Calculate trade-specific metrics"""
        if self.trades_df.empty:
            return {}

        # Average trade metrics
        avg_pnl = self.trades_df['pnl'].mean()
        avg_pnl_pct = self.trades_df['pnl_percent'].mean() * 100
        avg_hold_time = self.trades_df['hold_time_minutes'].mean()

        # Best and worst trades
        best_trade_pnl = self.trades_df['pnl'].max()
        worst_trade_pnl = self.trades_df['pnl'].min()

        # Average costs
        avg_transaction_costs = self.trades_df['transaction_costs'].mean()
        total_transaction_costs = self.trades_df['transaction_costs'].sum()

        return {
            'avg_trade_pnl': avg_pnl,
            'avg_trade_pnl_pct': avg_pnl_pct,
            'avg_hold_time_minutes': avg_hold_time,
            'avg_hold_time_hours': avg_hold_time / 60,
            'best_trade_pnl': best_trade_pnl,
            'worst_trade_pnl': worst_trade_pnl,
            'avg_transaction_costs': avg_transaction_costs,
            'total_transaction_costs': total_transaction_costs,
        }

    def _calculate_win_loss_metrics(self) -> Dict:
        """Calculate win/loss metrics"""
        if self.trades_df.empty:
            return {}

        # Winning and losing trades
        winning_trades = self.trades_df[self.trades_df['pnl'] > 0]
        losing_trades = self.trades_df[self.trades_df['pnl'] < 0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        total_trades = len(self.trades_df)

        win_rate = (num_wins / total_trades * 100) if total_trades > 0 else 0

        # Average win/loss
        avg_win = winning_trades['pnl'].mean() if num_wins > 0 else 0
        avg_loss = abs(losing_trades['pnl'].mean()) if num_losses > 0 else 0

        # Profit factor
        total_wins = winning_trades['pnl'].sum() if num_wins > 0 else 0
        total_losses = abs(losing_trades['pnl'].sum()) if num_losses > 0 else 0

        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        # Win/Loss ratio
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0

        # Expectancy
        expectancy = (win_rate / 100 * avg_win) - ((100 - win_rate) / 100 * avg_loss)

        # Consecutive wins/losses
        consecutive_wins = self._calculate_max_consecutive(self.trades_df, 'win')
        consecutive_losses = self._calculate_max_consecutive(self.trades_df, 'loss')

        return {
            'num_winning_trades': num_wins,
            'num_losing_trades': num_losses,
            'win_rate_pct': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'profit_factor': profit_factor,
            'win_loss_ratio': win_loss_ratio,
            'expectancy': expectancy,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
        }

    def _calculate_max_consecutive(self, trades_df: pd.DataFrame, result_type: str) -> int:
        """Calculate max consecutive wins or losses"""
        if trades_df.empty:
            return 0

        if result_type == 'win':
            results = (trades_df['pnl'] > 0).astype(int)
        else:  # loss
            results = (trades_df['pnl'] < 0).astype(int)

        max_consecutive = 0
        current_consecutive = 0

        for result in results:
            if result == 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def generate_monthly_returns(self) -> pd.DataFrame:
        """Generate monthly returns heatmap data"""
        if self.equity_curve.empty:
            return pd.DataFrame()

        # Set timestamp as index
        equity_df = self.equity_curve.set_index('timestamp')

        # Resample to monthly
        monthly_equity = equity_df['equity'].resample('M').last()

        # Calculate monthly returns
        monthly_returns = monthly_equity.pct_change() * 100

        # Create pivot table for heatmap
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })

        pivot = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')

        # Reorder months
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot.index = [month_names[i-1] for i in pivot.index]

        return pivot

    def save_metrics_to_json(self, filepath: str) -> None:
        """Save metrics to JSON file"""
        metrics = self.calculate_all_metrics()

        # Convert datetime objects to strings
        for key, value in metrics.items():
            if isinstance(value, (pd.Timestamp, datetime)):
                metrics[key] = value.isoformat()
            elif isinstance(value, (np.integer, np.floating)):
                metrics[key] = float(value)

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to {filepath}")


class ReportGenerator:
    """
    Generate HTML reports with interactive charts
    """

    def __init__(
        self,
        trades_df: pd.DataFrame,
        equity_curve: pd.DataFrame,
        metrics: Dict,
        strategy_config: Dict
    ):
        """Initialize report generator"""
        self.trades_df = trades_df
        self.equity_curve = equity_curve
        self.metrics = metrics
        self.strategy_config = strategy_config

        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available. Cannot generate HTML report.")

    def generate_html_report(self, output_path: str) -> None:
        """Generate comprehensive HTML report"""
        if not PLOTLY_AVAILABLE:
            logger.error("Cannot generate report: Plotly not installed")
            return

        logger.info("Generating HTML report...")

        # Create figures
        equity_fig = self._create_equity_chart()
        drawdown_fig = self._create_drawdown_chart()
        monthly_returns_fig = self._create_monthly_returns_heatmap()
        pnl_dist_fig = self._create_pnl_distribution()
        trade_duration_fig = self._create_trade_duration_chart()

        # Create HTML report
        html_content = self._create_html_content(
            equity_fig, drawdown_fig, monthly_returns_fig,
            pnl_dist_fig, trade_duration_fig
        )

        # Save to file
        with open(output_path, 'w') as f:
            f.write(html_content)

        logger.info(f"✅ HTML report saved to {output_path}")

    def _create_equity_chart(self) -> go.Figure:
        """Create equity curve chart"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.equity_curve['timestamp'],
            y=self.equity_curve['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='#2E86AB', width=2)
        ))

        fig.update_layout(
            title='Equity Curve',
            xaxis_title='Date',
            yaxis_title='Equity (₹)',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def _create_drawdown_chart(self) -> go.Figure:
        """Create drawdown chart"""
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.equity_curve['timestamp'],
            y=self.equity_curve['drawdown'],
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#A23B72', width=2)
        ))

        fig.update_layout(
            title='Drawdown (%)',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            hovermode='x unified',
            template='plotly_white'
        )

        return fig

    def _create_monthly_returns_heatmap(self) -> go.Figure:
        """Create monthly returns heatmap"""
        analytics = PerformanceAnalytics(
            self.trades_df, self.equity_curve,
            self.metrics['initial_capital']
        )

        monthly_returns = analytics.generate_monthly_returns()

        if monthly_returns.empty:
            # Return empty figure
            return go.Figure()

        fig = go.Figure(data=go.Heatmap(
            z=monthly_returns.values,
            x=monthly_returns.columns,
            y=monthly_returns.index,
            colorscale='RdYlGn',
            text=monthly_returns.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Return %")
        ))

        fig.update_layout(
            title='Monthly Returns Heatmap',
            xaxis_title='Year',
            yaxis_title='Month',
            template='plotly_white'
        )

        return fig

    def _create_pnl_distribution(self) -> go.Figure:
        """Create P&L distribution histogram"""
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=self.trades_df['pnl'],
            nbinsx=30,
            name='P&L Distribution',
            marker_color='#06A77D'
        ))

        fig.update_layout(
            title='Trade P&L Distribution',
            xaxis_title='P&L (₹)',
            yaxis_title='Frequency',
            template='plotly_white'
        )

        return fig

    def _create_trade_duration_chart(self) -> go.Figure:
        """Create trade duration chart"""
        fig = go.Figure()

        # Convert minutes to hours
        duration_hours = self.trades_df['hold_time_minutes'] / 60

        fig.add_trace(go.Histogram(
            x=duration_hours,
            nbinsx=30,
            name='Trade Duration',
            marker_color='#F18F01'
        ))

        fig.update_layout(
            title='Trade Duration Distribution',
            xaxis_title='Duration (hours)',
            yaxis_title='Frequency',
            template='plotly_white'
        )

        return fig

    def _create_html_content(
        self,
        equity_fig,
        drawdown_fig,
        monthly_returns_fig,
        pnl_dist_fig,
        trade_duration_fig
    ) -> str:
        """Create complete HTML content"""
        # Convert figures to HTML
        equity_html = equity_fig.to_html(full_html=False, include_plotlyjs='cdn')
        drawdown_html = drawdown_fig.to_html(full_html=False, include_plotlyjs=False)
        monthly_html = monthly_returns_fig.to_html(full_html=False, include_plotlyjs=False)
        pnl_html = pnl_dist_fig.to_html(full_html=False, include_plotlyjs=False)
        duration_html = trade_duration_fig.to_html(full_html=False, include_plotlyjs=False)

        # Create metrics table
        metrics_html = self._create_metrics_table()

        # Complete HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - BankNifty Options Strategy</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                .header {{
                    background-color: #2E86AB;
                    color: white;
                    padding: 20px;
                    text-align: center;
                    border-radius: 5px;
                }}
                .container {{
                    background-color: white;
                    padding: 20px;
                    margin: 20px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metrics-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .metrics-table th, .metrics-table td {{
                    padding: 12px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                .metrics-table th {{
                    background-color: #f8f9fa;
                    font-weight: bold;
                }}
                .positive {{
                    color: #06A77D;
                    font-weight: bold;
                }}
                .negative {{
                    color: #D62828;
                    font-weight: bold;
                }}
                .chart-container {{
                    margin: 30px 0;
                }}
                h2 {{
                    color: #2E86AB;
                    border-bottom: 2px solid #2E86AB;
                    padding-bottom: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BankNifty Options Strategy Backtest Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="container">
                <h2>Performance Summary</h2>
                {metrics_html}
            </div>

            <div class="container chart-container">
                <h2>Equity Curve</h2>
                {equity_html}
            </div>

            <div class="container chart-container">
                <h2>Drawdown</h2>
                {drawdown_html}
            </div>

            <div class="container chart-container">
                <h2>Monthly Returns</h2>
                {monthly_html}
            </div>

            <div class="container chart-container">
                <h2>P&L Distribution</h2>
                {pnl_html}
            </div>

            <div class="container chart-container">
                <h2>Trade Duration</h2>
                {duration_html}
            </div>
        </body>
        </html>
        """

        return html

    def _create_metrics_table(self) -> str:
        """Create HTML table for metrics"""
        def format_value(key, value):
            if value is None:
                return 'N/A'
            elif isinstance(value, (pd.Timestamp, datetime)):
                return value.strftime('%Y-%m-%d')
            elif isinstance(value, float):
                if 'pct' in key or 'rate' in key or 'return' in key or 'ratio' in key:
                    css_class = 'positive' if value > 0 else 'negative' if value < 0 else ''
                    return f'<span class="{css_class}">{value:.2f}%</span>' if 'pct' in key or 'rate' in key or 'return' in key else f'{value:.2f}'
                else:
                    return f'₹{value:,.2f}' if 'capital' in key or 'pnl' in key else f'{value:.2f}'
            else:
                return str(value)

        # Group metrics
        basic_metrics = ['initial_capital', 'final_capital', 'total_return_pct', 'cagr',
                        'start_date', 'end_date', 'duration_days', 'num_trades']

        risk_metrics = ['volatility_annual_pct', 'sharpe_ratio', 'sortino_ratio',
                       'calmar_ratio', 'max_drawdown_pct']

        trade_metrics = ['win_rate_pct', 'profit_factor', 'win_loss_ratio', 'expectancy',
                        'avg_trade_pnl', 'avg_hold_time_hours']

        html = '<table class="metrics-table">'

        # Basic metrics
        html += '<tr><th colspan="2" style="background-color: #2E86AB; color: white;">Basic Metrics</th></tr>'
        for key in basic_metrics:
            if key in self.metrics:
                label = key.replace('_', ' ').title()
                value = format_value(key, self.metrics[key])
                html += f'<tr><td>{label}</td><td>{value}</td></tr>'

        # Risk metrics
        html += '<tr><th colspan="2" style="background-color: #2E86AB; color: white;">Risk Metrics</th></tr>'
        for key in risk_metrics:
            if key in self.metrics:
                label = key.replace('_', ' ').title()
                value = format_value(key, self.metrics[key])
                html += f'<tr><td>{label}</td><td>{value}</td></tr>'

        # Trade metrics
        html += '<tr><th colspan="2" style="background-color: #2E86AB; color: white;">Trade Metrics</th></tr>'
        for key in trade_metrics:
            if key in self.metrics:
                label = key.replace('_', ' ').title()
                value = format_value(key, self.metrics[key])
                html += f'<tr><td>{label}</td><td>{value}</td></tr>'

        html += '</table>'

        return html


def analyze_backtest_results(
    trades_df: pd.DataFrame,
    equity_curve: pd.DataFrame,
    initial_capital: float,
    output_dir: str,
    strategy_config: Dict
) -> Dict:
    """
    Analyze backtest results and generate all outputs

    Args:
        trades_df: DataFrame with trades
        equity_curve: DataFrame with equity curve
        initial_capital: Initial capital
        output_dir: Output directory for reports
        strategy_config: Strategy configuration

    Returns:
        Dictionary with all metrics
    """
    logger.info("=" * 60)
    logger.info("Analyzing backtest results...")
    logger.info("=" * 60)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Calculate metrics
    analytics = PerformanceAnalytics(trades_df, equity_curve, initial_capital)
    metrics = analytics.calculate_all_metrics()

    # Save metrics to JSON
    metrics_path = f"{output_dir}/performance_metrics.json"
    analytics.save_metrics_to_json(metrics_path)

    # Save trades to CSV
    trades_path = f"{output_dir}/trades.csv"
    trades_df.to_csv(trades_path, index=False)
    logger.info(f"Trades saved to {trades_path}")

    # Save equity curve
    equity_path = f"{output_dir}/equity_curve.csv"
    equity_curve.to_csv(equity_path, index=False)
    logger.info(f"Equity curve saved to {equity_path}")

    # Generate HTML report
    if ANALYTICS_CONFIG['html_report']:
        report_gen = ReportGenerator(trades_df, equity_curve, metrics, strategy_config)
        report_path = f"{output_dir}/backtest_report.html"
        report_gen.generate_html_report(report_path)

    logger.info("=" * 60)
    logger.info("✅ Analysis completed!")
    logger.info("=" * 60)

    return metrics


if __name__ == "__main__":
    logger.info("Backtest analytics module loaded")
