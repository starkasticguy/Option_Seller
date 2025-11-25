"""
Monte Carlo Simulation Module
Run Monte Carlo simulations for strategy robustness testing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json

from backtest_config import MONTE_CARLO_CONFIG, STRATEGY_CONFIG, DATA_CONFIG
from backtest_engine import BacktestEngine
from backtest_analytics import PerformanceAnalytics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulator for backtesting parameter variations
    """

    def __init__(
        self,
        market_data: pd.DataFrame,
        base_strategy_config: Dict = None,
        mc_config: Dict = None
    ):
        """
        Initialize Monte Carlo simulator

        Args:
            market_data: Historical market data
            base_strategy_config: Base strategy configuration
            mc_config: Monte Carlo configuration
        """
        self.market_data = market_data
        self.base_strategy_config = base_strategy_config or STRATEGY_CONFIG
        self.mc_config = mc_config or MONTE_CARLO_CONFIG

        logger.info("Monte Carlo simulator initialized")
        logger.info(f"  Number of simulations: {self.mc_config['num_simulations']}")
        logger.info(f"  Vary parameters: {self.mc_config['vary_parameters']}")

    def run_parameter_variations(self) -> Dict:
        """
        Run simulations with parameter variations

        Returns:
            Dictionary with all simulation results
        """
        logger.info("=" * 60)
        logger.info("Running parameter variation simulations...")
        logger.info("=" * 60)

        if not self.mc_config['vary_parameters']:
            logger.info("Parameter variation disabled")
            return {}

        param_variations = self.mc_config['param_variations']

        # Generate all parameter combinations
        param_combinations = self._generate_parameter_combinations(param_variations)

        logger.info(f"Total parameter combinations: {len(param_combinations)}")

        # Run simulations
        results = []

        for idx, params in enumerate(param_combinations):
            logger.info(f"\nSimulation {idx+1}/{len(param_combinations)}")
            logger.info(f"  Parameters: {params}")

            # Create modified config
            config = deepcopy(self.base_strategy_config)
            for key, value in params.items():
                if key == 'vix_range':
                    config['vix_range'] = value
                elif key == 'profit_target':
                    config['profit_target'] = value
                elif key == 'stop_loss':
                    config['stop_loss'] = value
                elif key == 'delta_target':
                    config['ce_delta_target'] = -value
                    config['pe_delta_target'] = value

            # Run backtest
            try:
                engine = BacktestEngine(self.market_data, config)
                backtest_results = engine.run()

                # Calculate metrics
                trades_df = engine.get_trades_dataframe()
                equity_curve = backtest_results['equity_curve']

                if not trades_df.empty:
                    analytics = PerformanceAnalytics(
                        trades_df,
                        equity_curve,
                        backtest_results['initial_capital']
                    )
                    metrics = analytics.calculate_all_metrics()

                    results.append({
                        'parameters': params,
                        'metrics': metrics,
                        'num_trades': len(trades_df)
                    })
                else:
                    logger.warning("  No trades generated with these parameters")

            except Exception as e:
                logger.error(f"  Error in simulation: {str(e)}")
                continue

        # Analyze results
        analysis = self._analyze_parameter_variations(results)

        logger.info("=" * 60)
        logger.info("✅ Parameter variation completed!")
        logger.info(f"  Successful simulations: {len(results)}")
        logger.info("=" * 60)

        return {
            'all_results': results,
            'analysis': analysis
        }

    def run_regime_tests(self) -> Dict:
        """
        Test strategy in different market regimes

        Returns:
            Dictionary with regime test results
        """
        logger.info("=" * 60)
        logger.info("Running regime tests...")
        logger.info("=" * 60)

        if not self.mc_config['test_regimes']:
            logger.info("Regime testing disabled")
            return {}

        regimes = self.mc_config['regimes']
        results = {}

        for regime_name, regime_params in regimes.items():
            logger.info(f"\nTesting regime: {regime_name}")
            logger.info(f"  Parameters: {regime_params}")

            # Modify market data for regime
            modified_data = self._apply_regime_to_data(
                self.market_data,
                regime_params
            )

            # Run backtest
            try:
                engine = BacktestEngine(modified_data, self.base_strategy_config)
                backtest_results = engine.run()

                # Calculate metrics
                trades_df = engine.get_trades_dataframe()
                equity_curve = backtest_results['equity_curve']

                if not trades_df.empty:
                    analytics = PerformanceAnalytics(
                        trades_df,
                        equity_curve,
                        backtest_results['initial_capital']
                    )
                    metrics = analytics.calculate_all_metrics()

                    results[regime_name] = {
                        'regime_params': regime_params,
                        'metrics': metrics,
                        'num_trades': len(trades_df)
                    }

                    logger.info(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
                    logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")

            except Exception as e:
                logger.error(f"  Error in regime test: {str(e)}")
                continue

        logger.info("=" * 60)
        logger.info("✅ Regime testing completed!")
        logger.info("=" * 60)

        return results

    def run_stress_tests(self) -> Dict:
        """
        Run stress tests with extreme scenarios

        Returns:
            Dictionary with stress test results
        """
        logger.info("=" * 60)
        logger.info("Running stress tests...")
        logger.info("=" * 60)

        if not self.mc_config['stress_test']:
            logger.info("Stress testing disabled")
            return {}

        stress_scenarios = self.mc_config['stress_scenarios']
        results = {}

        for scenario_name, scenario_params in stress_scenarios.items():
            logger.info(f"\nTesting scenario: {scenario_name}")
            logger.info(f"  Parameters: {scenario_params}")

            # Modify market data for stress scenario
            modified_data = self._apply_stress_scenario(
                self.market_data,
                scenario_params
            )

            # Run backtest
            try:
                engine = BacktestEngine(modified_data, self.base_strategy_config)
                backtest_results = engine.run()

                # Calculate metrics
                trades_df = engine.get_trades_dataframe()
                equity_curve = backtest_results['equity_curve']

                if not trades_df.empty:
                    analytics = PerformanceAnalytics(
                        trades_df,
                        equity_curve,
                        backtest_results['initial_capital']
                    )
                    metrics = analytics.calculate_all_metrics()

                    results[scenario_name] = {
                        'scenario_params': scenario_params,
                        'metrics': metrics,
                        'num_trades': len(trades_df)
                    }

                    logger.info(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
                    logger.info(f"  Max Drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")

            except Exception as e:
                logger.error(f"  Error in stress test: {str(e)}")
                continue

        logger.info("=" * 60)
        logger.info("✅ Stress testing completed!")
        logger.info("=" * 60)

        return results

    def run_bootstrap_simulations(
        self,
        num_simulations: int = None
    ) -> Dict:
        """
        Run bootstrap simulations by resampling trades

        Args:
            num_simulations: Number of bootstrap simulations

        Returns:
            Dictionary with bootstrap results
        """
        logger.info("=" * 60)
        logger.info("Running bootstrap simulations...")
        logger.info("=" * 60)

        num_simulations = num_simulations or self.mc_config['num_simulations']

        # First run base backtest
        engine = BacktestEngine(self.market_data, self.base_strategy_config)
        backtest_results = engine.run()

        trades_df = engine.get_trades_dataframe()

        if trades_df.empty:
            logger.warning("No trades in base backtest")
            return {}

        # Run bootstrap simulations
        bootstrap_results = []

        for sim_num in range(num_simulations):
            if sim_num % 100 == 0:
                logger.info(f"  Simulation {sim_num}/{num_simulations}")

            # Resample trades with replacement
            resampled_trades = trades_df.sample(n=len(trades_df), replace=True)

            # Calculate equity curve from resampled trades
            initial_capital = backtest_results['initial_capital']
            equity = initial_capital
            equity_curve_data = []

            for _, trade in resampled_trades.iterrows():
                equity += trade['pnl']
                equity_curve_data.append(equity)

            # Calculate metrics
            final_equity = equity
            total_return = (final_equity / initial_capital - 1) * 100

            # Calculate Sharpe ratio from trade returns
            trade_returns = resampled_trades['pnl_percent']
            sharpe = (trade_returns.mean() / trade_returns.std()) * np.sqrt(252) if trade_returns.std() > 0 else 0

            # Calculate max drawdown
            equity_series = pd.Series(equity_curve_data)
            peak = equity_series.expanding(min_periods=1).max()
            drawdown = (equity_series / peak - 1) * 100
            max_dd = drawdown.min()

            bootstrap_results.append({
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown_pct': max_dd,
                'num_wins': len(resampled_trades[resampled_trades['pnl'] > 0]),
                'win_rate': len(resampled_trades[resampled_trades['pnl'] > 0]) / len(resampled_trades) * 100
            })

        # Calculate confidence intervals
        confidence_levels = self.mc_config['confidence_levels']
        confidence_intervals = self._calculate_confidence_intervals(
            bootstrap_results,
            confidence_levels
        )

        logger.info("=" * 60)
        logger.info("✅ Bootstrap simulations completed!")
        logger.info(f"  Total simulations: {num_simulations}")
        logger.info("=" * 60)

        return {
            'bootstrap_results': bootstrap_results,
            'confidence_intervals': confidence_intervals
        }

    def _generate_parameter_combinations(
        self,
        param_variations: Dict
    ) -> List[Dict]:
        """Generate all parameter combinations"""
        import itertools

        # Get all parameter names and values
        param_names = list(param_variations.keys())
        param_values = [param_variations[name] for name in param_names]

        # Generate all combinations
        combinations = list(itertools.product(*param_values))

        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = {name: value for name, value in zip(param_names, combo)}
            param_combinations.append(param_dict)

        return param_combinations

    def _apply_regime_to_data(
        self,
        data: pd.DataFrame,
        regime_params: Dict
    ) -> pd.DataFrame:
        """Apply regime modifications to market data"""
        modified_data = data.copy()

        # Apply drift
        drift = regime_params.get('drift', 0)
        if drift != 0:
            # Add drift to returns
            returns = modified_data['Close'].pct_change()
            adjusted_returns = returns + drift
            modified_data['Close'] = modified_data['Close'].iloc[0] * (1 + adjusted_returns).cumprod()

            # Recalculate OHLC
            modified_data['High'] = modified_data['Close'] * 1.01
            modified_data['Low'] = modified_data['Close'] * 0.99
            modified_data['Open'] = modified_data['Close'].shift(1)

        # Apply volatility multiplier
        vol_mult = regime_params.get('volatility_mult', 1.0)
        if vol_mult != 1.0:
            # Scale VIX
            if 'VIX' in modified_data.columns:
                modified_data['VIX'] = modified_data['VIX'] * vol_mult

            # Scale realized volatility
            for col in modified_data.columns:
                if col.startswith('RV_'):
                    modified_data[col] = modified_data[col] * vol_mult

        return modified_data

    def _apply_stress_scenario(
        self,
        data: pd.DataFrame,
        scenario_params: Dict
    ) -> pd.DataFrame:
        """Apply stress scenario to market data"""
        modified_data = data.copy()

        # Apply sudden gaps
        if 'gap_percent' in scenario_params:
            gap_pct = scenario_params['gap_percent']
            frequency = scenario_params.get('frequency', 0.01)

            # Randomly apply gaps
            num_gaps = int(len(modified_data) * frequency)
            gap_indices = np.random.choice(len(modified_data), num_gaps, replace=False)

            for idx in gap_indices:
                if idx > 0:
                    # Apply gap
                    gap_direction = np.random.choice([-1, 1])
                    gap_multiplier = 1 + (gap_direction * gap_pct)

                    modified_data.loc[modified_data.index[idx]:, 'Close'] *= gap_multiplier
                    modified_data.loc[modified_data.index[idx]:, 'High'] *= gap_multiplier
                    modified_data.loc[modified_data.index[idx]:, 'Low'] *= gap_multiplier

        # Apply volatility spikes
        if 'vol_mult' in scenario_params:
            vol_mult = scenario_params['vol_mult']
            frequency = scenario_params.get('frequency', 0.05)

            # Randomly spike VIX
            num_spikes = int(len(modified_data) * frequency)
            spike_indices = np.random.choice(len(modified_data), num_spikes, replace=False)

            if 'VIX' in modified_data.columns:
                for idx in spike_indices:
                    # Spike VIX
                    spike_duration = np.random.randint(5, 20)  # 5-20 bars
                    end_idx = min(idx + spike_duration, len(modified_data))

                    modified_data.loc[
                        modified_data.index[idx]:modified_data.index[end_idx-1],
                        'VIX'
                    ] *= vol_mult

        return modified_data

    def _analyze_parameter_variations(
        self,
        results: List[Dict]
    ) -> Dict:
        """Analyze parameter variation results"""
        if not results:
            return {}

        # Extract metrics
        returns = [r['metrics'].get('total_return_pct', 0) for r in results]
        sharpes = [r['metrics'].get('sharpe_ratio', 0) for r in results]
        max_dds = [r['metrics'].get('max_drawdown_pct', 0) for r in results]

        # Find best configurations
        best_return_idx = np.argmax(returns)
        best_sharpe_idx = np.argmax(sharpes)

        analysis = {
            'num_simulations': len(results),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_sharpe': np.mean(sharpes),
            'std_sharpe': np.std(sharpes),
            'avg_max_dd': np.mean(max_dds),
            'best_return_params': results[best_return_idx]['parameters'],
            'best_return_value': returns[best_return_idx],
            'best_sharpe_params': results[best_sharpe_idx]['parameters'],
            'best_sharpe_value': sharpes[best_sharpe_idx],
        }

        return analysis

    def _calculate_confidence_intervals(
        self,
        bootstrap_results: List[Dict],
        confidence_levels: List[float]
    ) -> Dict:
        """Calculate confidence intervals from bootstrap results"""
        intervals = {}

        # Extract metrics
        metrics_names = list(bootstrap_results[0].keys())

        for metric in metrics_names:
            values = [r[metric] for r in bootstrap_results]

            metric_intervals = {}
            for conf_level in confidence_levels:
                # Calculate percentiles
                lower_percentile = (1 - conf_level) / 2 * 100
                upper_percentile = (1 + conf_level) / 2 * 100

                lower = np.percentile(values, lower_percentile)
                upper = np.percentile(values, upper_percentile)

                metric_intervals[f'{int(conf_level*100)}%'] = {
                    'lower': lower,
                    'upper': upper,
                    'mean': np.mean(values),
                    'median': np.median(values)
                }

            intervals[metric] = metric_intervals

        return intervals

    def save_results(self, results: Dict, output_path: str) -> None:
        """Save Monte Carlo results to JSON"""
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            else:
                return obj

        results_converted = convert_types(results)

        with open(output_path, 'w') as f:
            json.dump(results_converted, f, indent=2)

        logger.info(f"Monte Carlo results saved to {output_path}")


def run_full_monte_carlo(
    market_data: pd.DataFrame,
    strategy_config: Dict,
    output_dir: str
) -> Dict:
    """
    Run full Monte Carlo analysis

    Args:
        market_data: Historical market data
        strategy_config: Strategy configuration
        output_dir: Output directory

    Returns:
        Dictionary with all results
    """
    logger.info("=" * 60)
    logger.info("Starting Full Monte Carlo Analysis")
    logger.info("=" * 60)

    simulator = MonteCarloSimulator(market_data, strategy_config)

    results = {}

    # Parameter variations
    if MONTE_CARLO_CONFIG['vary_parameters']:
        results['parameter_variations'] = simulator.run_parameter_variations()

    # Regime tests
    if MONTE_CARLO_CONFIG['test_regimes']:
        results['regime_tests'] = simulator.run_regime_tests()

    # Stress tests
    if MONTE_CARLO_CONFIG['stress_test']:
        results['stress_tests'] = simulator.run_stress_tests()

    # Bootstrap simulations
    results['bootstrap'] = simulator.run_bootstrap_simulations()

    # Save results
    output_path = f"{output_dir}/monte_carlo_results.json"
    simulator.save_results(results, output_path)

    logger.info("=" * 60)
    logger.info("✅ Full Monte Carlo Analysis Completed!")
    logger.info("=" * 60)

    return results


if __name__ == "__main__":
    logger.info("Monte Carlo simulation module loaded")
