#!/usr/bin/env python3
"""
Portfolio Optimization Framework
================================

Implementation of Modern Portfolio Theory (MPT) with multiple optimization strategies:
- Mean-Variance Optimization (Markowitz)
- Risk Parity
- Black-Litterman Model
- Monte Carlo Simulation
- Efficient Frontier Analysis

Author: Benggoy
License: MIT
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy import linalg
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    Comprehensive Portfolio Optimization Framework
    
    Features:
    - Multiple optimization methods
    - Risk metrics calculation
    - Efficient frontier generation
    - Monte Carlo simulation
    - Performance analytics
    """
    
    def __init__(self, tickers: List[str], start_date: str = None, end_date: str = None):
        """
        Initialize Portfolio Optimizer
        
        Args:
            tickers (List[str]): List of stock tickers
            start_date (str): Start date for data (YYYY-MM-DD)
            end_date (str): End date for data (YYYY-MM-DD)
        """
        self.tickers = tickers
        self.start_date = start_date or (datetime.now() - timedelta(days=252*2)).strftime('%Y-%m-%d')
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Data containers
        self.prices = None
        self.returns = None
        self.cov_matrix = None
        self.mean_returns = None
        
        # Load and process data
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess stock data"""
        print(f"Loading data for {len(self.tickers)} assets...")
        
        try:
            # Download price data
            data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
            
            if len(self.tickers) == 1:
                data = data.to_frame()
                data.columns = self.tickers
                
            # Handle missing data
            data = data.dropna()
            
            if data.empty:
                raise ValueError("No valid data found for the given tickers and date range")
                
            self.prices = data
            
            # Calculate returns
            self.returns = data.pct_change().dropna()
            
            # Calculate statistics
            self.mean_returns = self.returns.mean() * 252  # Annualized
            self.cov_matrix = self.returns.cov() * 252     # Annualized
            
            print(f"✅ Data loaded successfully")
            print(f"   Period: {self.prices.index[0].strftime('%Y-%m-%d')} to {self.prices.index[-1].strftime('%Y-%m-%d')}")
            print(f"   Assets: {list(self.prices.columns)}")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics
        
        Args:
            weights (np.ndarray): Portfolio weights
            
        Returns:
            Tuple[float, float, float]: (returns, volatility, sharpe_ratio)
        """
        # Portfolio return
        portfolio_return = np.sum(self.mean_returns * weights)
        
        # Portfolio volatility
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def optimize_sharpe_ratio(self) -> Dict:
        """
        Optimize portfolio for maximum Sharpe ratio
        
        Returns:
            Dict: Optimization results
        """
        num_assets = len(self.tickers)
        
        # Objective function (negative Sharpe ratio to minimize)
        def negative_sharpe(weights):
            return -self.portfolio_performance(weights)[2]
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
        bounds = tuple((0, 1) for _ in range(num_assets))  # Long-only
        
        # Initial guess (equal weights)
        initial_guess = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_performance(optimal_weights)
            
            return {
                'method': 'Maximum Sharpe Ratio',
                'weights': dict(zip(self.tickers, optimal_weights)),
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': 'Optimization failed'}
    
    def optimize_min_volatility(self) -> Dict:
        """
        Optimize portfolio for minimum volatility
        
        Returns:
            Dict: Optimization results
        """
        num_assets = len(self.tickers)
        
        # Objective function
        def portfolio_volatility(weights):
            return self.portfolio_performance(weights)[1]
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess
        initial_guess = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_performance(optimal_weights)
            
            return {
                'method': 'Minimum Volatility',
                'weights': dict(zip(self.tickers, optimal_weights)),
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': 'Optimization failed'}
    
    def risk_parity_optimization(self) -> Dict:
        """
        Risk Parity optimization - equal risk contribution
        
        Returns:
            Dict: Optimization results
        """
        num_assets = len(self.tickers)
        
        def risk_budget_objective(weights):
            """Minimize difference between risk contributions"""
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            marginal_contrib = np.dot(self.cov_matrix, weights) / portfolio_vol
            contrib = weights * marginal_contrib
            target_contrib = 1.0 / num_assets
            return np.sum((contrib - target_contrib * portfolio_vol)**2)
        
        # Constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.01, 1) for _ in range(num_assets))  # Minimum 1% allocation
        
        # Initial guess
        initial_guess = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(risk_budget_objective, initial_guess, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            ret, vol, sharpe = self.portfolio_performance(optimal_weights)
            
            return {
                'method': 'Risk Parity',
                'weights': dict(zip(self.tickers, optimal_weights)),
                'expected_return': ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'success': True
            }
        else:
            return {'success': False, 'message': 'Risk parity optimization failed'}
    
    def efficient_frontier(self, num_portfolios: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier
        
        Args:
            num_portfolios (int): Number of portfolios to generate
            
        Returns:
            pd.DataFrame: Efficient frontier data
        """
        num_assets = len(self.tickers)
        
        # Get min and max returns
        min_vol_result = self.optimize_min_volatility()
        max_sharpe_result = self.optimize_sharpe_ratio()
        
        min_return = min_vol_result['expected_return']
        max_return = max_sharpe_result['expected_return']
        
        target_returns = np.linspace(min_return, max_return, num_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # Minimize volatility for target return
            def portfolio_volatility(weights):
                return self.portfolio_performance(weights)[1]
            
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: self.portfolio_performance(x)[0] - target_return}  # Target return
            ]
            bounds = tuple((0, 1) for _ in range(num_assets))
            initial_guess = np.array([1/num_assets] * num_assets)
            
            result = minimize(portfolio_volatility, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                ret, vol, sharpe = self.portfolio_performance(weights)
                
                portfolio_data = {'Return': ret, 'Volatility': vol, 'Sharpe_Ratio': sharpe}
                for i, ticker in enumerate(self.tickers):
                    portfolio_data[ticker] = weights[i]
                
                efficient_portfolios.append(portfolio_data)
        
        return pd.DataFrame(efficient_portfolios)
    
    def monte_carlo_simulation(self, num_simulations: int = 10000) -> pd.DataFrame:
        """
        Monte Carlo simulation for random portfolio generation
        
        Args:
            num_simulations (int): Number of random portfolios to generate
            
        Returns:
            pd.DataFrame: Simulation results
        """
        num_assets = len(self.tickers)
        results = []
        
        for _ in range(num_simulations):
            # Generate random weights
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)  # Normalize to sum to 1
            
            # Calculate performance
            ret, vol, sharpe = self.portfolio_performance(weights)
            
            # Store results
            portfolio_data = {'Return': ret, 'Volatility': vol, 'Sharpe_Ratio': sharpe}
            for i, ticker in enumerate(self.tickers):
                portfolio_data[ticker] = weights[i]
            
            results.append(portfolio_data)
        
        return pd.DataFrame(results)
    
    def plot_efficient_frontier(self, include_monte_carlo: bool = True):
        """
        Plot efficient frontier with optimization results
        
        Args:
            include_monte_carlo (bool): Include Monte Carlo simulation
        """
        # Generate efficient frontier
        print("Generating efficient frontier...")
        efficient_frontier_data = self.efficient_frontier()
        
        # Get optimization results
        max_sharpe = self.optimize_sharpe_ratio()
        min_vol = self.optimize_min_volatility()
        risk_parity = self.risk_parity_optimization()
        
        # Create plot
        fig = go.Figure()
        
        # Monte Carlo simulation
        if include_monte_carlo:
            print("Running Monte Carlo simulation...")
            mc_data = self.monte_carlo_simulation(5000)
            
            fig.add_trace(go.Scatter(
                x=mc_data['Volatility'],
                y=mc_data['Return'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=mc_data['Sharpe_Ratio'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Sharpe Ratio"),
                    opacity=0.6
                ),
                name='Random Portfolios',
                hovertemplate='Return: %{y:.2%}<br>Volatility: %{x:.2%}<br>Sharpe: %{marker.color:.3f}'
            ))
        
        # Efficient frontier
        fig.add_trace(go.Scatter(
            x=efficient_frontier_data['Volatility'],
            y=efficient_frontier_data['Return'],
            mode='lines',
            line=dict(color='red', width=3),
            name='Efficient Frontier'
        ))
        
        # Optimal portfolios
        if max_sharpe['success']:
            fig.add_trace(go.Scatter(
                x=[max_sharpe['volatility']],
                y=[max_sharpe['expected_return']],
                mode='markers',
                marker=dict(color='gold', size=15, symbol='star'),
                name='Max Sharpe Ratio',
                hovertemplate=f"Max Sharpe<br>Return: {max_sharpe['expected_return']:.2%}<br>Volatility: {max_sharpe['volatility']:.2%}<br>Sharpe: {max_sharpe['sharpe_ratio']:.3f}"
            ))
        
        if min_vol['success']:
            fig.add_trace(go.Scatter(
                x=[min_vol['volatility']],
                y=[min_vol['expected_return']],
                mode='markers',
                marker=dict(color='green', size=15, symbol='diamond'),
                name='Min Volatility',
                hovertemplate=f"Min Volatility<br>Return: {min_vol['expected_return']:.2%}<br>Volatility: {min_vol['volatility']:.2%}<br>Sharpe: {min_vol['sharpe_ratio']:.3f}"
            ))
        
        if risk_parity['success']:
            fig.add_trace(go.Scatter(
                x=[risk_parity['volatility']],
                y=[risk_parity['expected_return']],
                mode='markers',
                marker=dict(color='blue', size=15, symbol='square'),
                name='Risk Parity',
                hovertemplate=f"Risk Parity<br>Return: {risk_parity['expected_return']:.2%}<br>Volatility: {risk_parity['volatility']:.2%}<br>Sharpe: {risk_parity['sharpe_ratio']:.3f}"
            ))
        
        # Layout
        fig.update_layout(
            title=f'Portfolio Optimization - Efficient Frontier<br>Assets: {", ".join(self.tickers)}',
            xaxis_title='Volatility (Risk)',
            yaxis_title='Expected Return',
            width=1000,
            height=700,
            showlegend=True
        )
        
        # Format axes as percentages
        fig.update_xaxes(tickformat='.1%')
        fig.update_yaxes(tickformat='.1%')
        
        fig.show()
    
    def generate_report(self) -> None:
        """Generate comprehensive portfolio optimization report"""
        
        print("="*80)
        print("📊 PORTFOLIO OPTIMIZATION REPORT")
        print("="*80)
        
        # Basic information
        print(f"\n🎯 ANALYSIS OVERVIEW:")
        print(f"Assets: {', '.join(self.tickers)}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Observations: {len(self.returns)} days")
        
        # Individual asset statistics
        print(f"\n📈 INDIVIDUAL ASSET STATISTICS:")
        stats_df = pd.DataFrame({
            'Expected Return': self.mean_returns,
            'Volatility': np.sqrt(np.diag(self.cov_matrix)),
            'Sharpe Ratio': (self.mean_returns - 0.02) / np.sqrt(np.diag(self.cov_matrix))
        })
        print(stats_df.round(4))
        
        # Correlation matrix
        print(f"\n🔗 CORRELATION MATRIX:")
        corr_matrix = self.returns.corr()
        print(corr_matrix.round(3))
        
        # Optimization results
        print(f"\n🎯 OPTIMIZATION RESULTS:")
        print("-" * 60)
        
        optimizations = [
            self.optimize_sharpe_ratio(),
            self.optimize_min_volatility(),
            self.risk_parity_optimization()
        ]
        
        for opt_result in optimizations:
            if opt_result['success']:
                print(f"\n{opt_result['method'].upper()}:")
                print(f"Expected Return: {opt_result['expected_return']:.2%}")
                print(f"Volatility: {opt_result['volatility']:.2%}")
                print(f"Sharpe Ratio: {opt_result['sharpe_ratio']:.3f}")
                print("Asset Allocation:")
                for ticker, weight in opt_result['weights'].items():
                    print(f"  {ticker}: {weight:.1%}")
        
        # Performance comparison
        print(f"\n📊 PERFORMANCE COMPARISON:")
        comparison_data = []
        for opt_result in optimizations:
            if opt_result['success']:
                comparison_data.append({
                    'Strategy': opt_result['method'],
                    'Return': f"{opt_result['expected_return']:.2%}",
                    'Volatility': f"{opt_result['volatility']:.2%}",
                    'Sharpe': f"{opt_result['sharpe_ratio']:.3f}"
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        print(f"\n{'='*80}")
        print("📈 Use plot_efficient_frontier() to visualize results")
        print("="*80)

def example_usage():
    """Example usage of Portfolio Optimizer"""
    
    # Example 1: Tech stocks
    print("🚀 PORTFOLIO OPTIMIZATION EXAMPLE")
    print("="*50)
    
    # Define portfolio
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    try:
        # Initialize optimizer
        optimizer = PortfolioOptimizer(
            tickers=tech_stocks,
            start_date='2022-01-01',
            end_date='2024-12-31'
        )
        
        # Generate comprehensive report
        optimizer.generate_report()
        
        # Plot efficient frontier
        print("\n📊 Generating efficient frontier visualization...")
        optimizer.plot_efficient_frontier(include_monte_carlo=True)
        
    except Exception as e:
        print(f"❌ Error in example: {e}")

if __name__ == "__main__":
    example_usage()
