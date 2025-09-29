#!/usr/bin/env python3
"""
Portfolio Rebalancing Tools for LangGraph ReAct Agent
This module provides tools for portfolio analysis and rebalancing against target indices.
"""

import json
import pandas as pd
from langchain_core.tools import tool
from typing import Dict, Any, List
from finance.tools.portfolio_data import validate_index_name
from finance.tools.portfolio_data import get_index_weights, get_prices, INDEX_WEIGHTS, STOCK_PRICES

# Dictionary Validation Functions
def validate_portfolio_holdings(holdings: Dict[str, float]) -> Dict[str, float]:
    """Validate portfolio holdings dictionary."""
    if not holdings:
        raise ValueError("Portfolio must have at least one holding")
    
    validated_holdings = {}
    for ticker, quantity in holdings.items():
        if not isinstance(ticker, str) or len(ticker.strip()) == 0:
            raise ValueError(f"Invalid ticker: {ticker}")
        if not isinstance(quantity, (int, float)) or quantity < 0:
            raise ValueError(f"Invalid quantity for {ticker}: {quantity}")
        validated_holdings[ticker.upper().strip()] = float(quantity)
    
    return validated_holdings



def validate_rebalancing_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Validate rebalancing request dictionary."""
    if "portfolio" not in request or "holdings" not in request["portfolio"]:
        raise ValueError("Request must contain portfolio.holdings")
    if "index_name" not in request:
        raise ValueError("Request must contain index_name")
    
    validated_request = {
        "portfolio": {"holdings": validate_portfolio_holdings(request["portfolio"]["holdings"])},
        "index_name": validate_index_name(request["index_name"]),
        "lot_size": float(request.get("lot_size", 1.0))
    }
    
    if validated_request["lot_size"] <= 0:
        raise ValueError("lot_size must be greater than 0")
    
    return validated_request

def create_portfolio_holding(ticker: str, shares: float, price: float, market_value: float, weight_percent: float) -> Dict[str, Any]:
    """Create portfolio holding dictionary."""
    return {
        "ticker": ticker,
        "shares": shares,
        "price": price,
        "market_value": market_value,
        "weight_percent": weight_percent
    }

def create_rebalancing_action(ticker: str, action: str, current_weight_percent: float,
                             target_weight_percent: float, shares: float, amount: float) -> Dict[str, Any]:
    """Create rebalancing action dictionary."""
    return {
        "ticker": ticker,
        "action": action,
        "current_weight_percent": current_weight_percent,
        "target_weight_percent": target_weight_percent,
        "shares": shares,
        "amount": amount
    }

def create_rebalancing_result(target_index: str, current_portfolio_value: float, lot_size: float,
                             current_holdings: List[Dict], rebalancing_actions: List[Dict],
                             target_holdings: List[Dict], total_trading_value: float,
                             number_of_actions: int) -> Dict[str, Any]:
    """Create rebalancing result dictionary."""
    return {
        "target_index": target_index,
        "current_portfolio_value": current_portfolio_value,
        "lot_size": lot_size,
        "current_holdings": current_holdings,
        "rebalancing_actions": rebalancing_actions,
        "target_holdings": target_holdings,
        "total_trading_value": total_trading_value,
        "number_of_actions": number_of_actions
    }

def format_rebalancing_summary(result: Dict[str, Any]) -> str:
    """Format the rebalancing result as a human-readable string."""
    output = f"🔄 Portfolio Rebalancing Analysis\n"
    output += "=" * 50 + "\n"
    output += f"Target Index: {result['target_index'].upper()}\n"
    output += f"Current Portfolio Value: ${result['current_portfolio_value']:,.2f}\n"
    output += f"Lot Size: {result['lot_size']} shares per lot\n\n"
    
    # Current Portfolio Analysis
    output += "📈 Current Portfolio:\n"
    output += "-" * 30 + "\n"
    for holding in result['current_holdings']:
        output += f"{holding['ticker']:<6}: {holding['shares']:>8.0f} shares @ ${holding['price']:>7.2f} = ${holding['market_value']:>10.2f} ({holding['weight_percent']:>5.1f}%)\n"
    
    # Target Analysis & Rebalancing
    output += f"\n🎯 Target Allocation & Rebalancing:\n"
    output += "-" * 50 + "\n"
    output += f"{'Ticker':<6} {'Current':<8} {'Target':<8} {'Action':<12} {'Shares':<8} {'Amount':<12}\n"
    output += "-" * 50 + "\n"
    
    for action in result['rebalancing_actions']:
        if action['action'] != "HOLD":
            output += f"{action['ticker']:<6} {action['current_weight_percent']:>6.1f}% {action['target_weight_percent']:>6.1f}% "
            output += f"{action['action']:<12} {action['shares']:>8.2f} ${action['amount']:>10.2f}\n"
        else:
            output += f"{action['ticker']:<6} {action['current_weight_percent']:>6.1f}% {action['target_weight_percent']:>6.1f}% {'HOLD':<12} {'0':<8} ${'0.00':>10}\n"
    
    # Summary
    output += "-" * 50 + "\n"
    output += f"Total Trading Value: ${result['total_trading_value']:,.2f}\n"
    output += f"Number of Actions: {result['number_of_actions']}\n\n"
    
    # Final target portfolio
    output += "🏁 Final Target Portfolio:\n"
    output += "-" * 30 + "\n"
    for holding in result['target_holdings']:
        output += f"{holding['ticker']:<6}: {holding['shares']:>8.2f} shares = ${holding['market_value']:>10.2f} ({holding['weight_percent']:>5.1f}%)\n"
    
    return output


@tool
def rebalance_portfolio(holdings: Dict[str, float], index_name: str, lot_size: float = 1) -> Dict[str, Any]:
    """
    Rebalance a portfolio to match target index weights using pandas for efficient calculations.
    
    This function analyzes a current portfolio and provides detailed rebalancing recommendations
    to match a target index allocation, considering minimum lot size constraints.
    
    Args:
        holdings (Dict[str, float]): Dictionary mapping ticker symbols to number of shares.
            - Keys: Stock ticker symbols as strings (e.g., "AAPL", "GOOGL")
            - Values: Number of shares as float/int (e.g., 100.0, 50)
            - Example: {"AAPL": 100.0, "GOOGL": 25.0, "MSFT": 75.0}
            - Validation: All tickers converted to uppercase, shares must be >= 0
            
        index_name (str): Name of the target index to rebalance towards.
            - Valid values: "index1" or "index2" (case-insensitive)
            - "index1": 5-stock tech portfolio (AAPL, GOOGL, MSFT, AMZN, TSLA)
            - "index2": 10-stock diversified tech portfolio
            
        lot_size (float): Minimum trading lot size in shares per transaction.
            - Must be > 0.0
            - Example: 10.0 means trades must be in multiples of 10 shares
            - Trades below lot_size threshold will result in "HOLD" action
    
    Returns:
        Dict[str, Any]: Comprehensive rebalancing analysis with the following structure:
        
        {
            "target_index": str,                    # Target index name (validated)
            "current_portfolio_value": float,       # Total market value of current portfolio
            "lot_size": float,                     # Lot size used for calculations
            
            "current_holdings": List[Dict[str, Any]], # Current portfolio positions
            # Each holding dictionary contains:
            # {
            #     "ticker": str,           # Stock symbol (e.g., "AAPL")
            #     "shares": float,         # Current number of shares
            #     "price": float,          # Current stock price per share
            #     "market_value": float,   # shares * price
            #     "weight_percent": float  # Position weight as percentage (0-100)
            # }
            
            "rebalancing_actions": List[Dict[str, Any]], # Recommended trading actions
            # Each action dictionary contains:
            # {
            #     "ticker": str,                    # Stock symbol
            #     "action": str,                    # "BUY", "SELL", or "HOLD"
            #     "current_weight_percent": float,  # Current allocation percentage
            #     "target_weight_percent": float,   # Target allocation percentage
            #     "shares": float,                  # Number of shares to trade (0 if HOLD)
            #     "amount": float                   # Dollar amount of trade (0 if HOLD)
            # }
            
            "target_holdings": List[Dict[str, Any]],  # Final portfolio after rebalancing
            # Each target holding has same structure as current_holdings above
            
            "total_trading_value": float,        # Total dollar value of all trades
            "number_of_actions": int            # Count of non-HOLD actions required
        }
    
    Example Usage:
        >>> holdings = {"AAPL": 100, "GOOGL": 20, "MSFT": 50}
        >>> result = rebalance_portfolio(holdings, "index1", 10.0)
        >>> print(f"Portfolio value: ${result['current_portfolio_value']:,.2f}")
        >>> print(f"Actions needed: {result['number_of_actions']}")
        >>> for action in result['rebalancing_actions']:
        ...     if action['action'] != 'HOLD':
        ...         print(f"{action['action']} {action['shares']} shares of {action['ticker']}")
    
    Raises:
        ValueError: If invalid input parameters are provided:
            - Empty or invalid holdings dictionary
            - Invalid index_name (must be 'index1' or 'index2')
            - Invalid lot_size (must be > 0)
            - Missing price data for any ticker
            
    Notes:
        - All ticker symbols are automatically converted to uppercase
        - Stock prices are sourced from hardcoded STOCK_PRICES dictionary
        - Rebalancing uses lot-based trading (trades rounded down to lot_size multiples)
        - Portfolio value calculation includes all current holdings
        - Target weights are defined in INDEX_WEIGHTS dictionary
        - Function returns structured data suitable for programmatic processing
    """
    try:
        print('holdings:', holdings)
        print('index_name:', index_name)
        print('lot_size:', lot_size)
        
        # Validate input parameters directly
        portfolio = validate_portfolio_holdings(holdings)
        validated_index_name = validate_index_name(index_name)
        validated_lot_size = float(lot_size)
        
        

        if validated_lot_size <= 0:
            raise ValueError("lot_size must be greater than 0")
            
        target_weights = INDEX_WEIGHTS[validated_index_name]
        
        # Create comprehensive DataFrame with all tickers
        all_tickers = set(list(portfolio.keys()) + list(target_weights.keys()))
        
        # Build portfolio DataFrame
        df = pd.DataFrame({
            'ticker': [ticker.upper() for ticker in all_tickers],
            'current_shares': [portfolio.get(ticker, portfolio.get(ticker.upper(), 0)) for ticker in all_tickers],
            'target_weight': [target_weights.get(ticker.upper(), 0) for ticker in all_tickers],
            'price': [STOCK_PRICES.get(ticker.upper(), 0) for ticker in all_tickers]
        })
        print('Original Portfolio Holdings & Target Weights DataFrame:')
        print(df)
        
        # Validate price data availability
        missing_prices = df[df['price'] == 0]['ticker'].tolist()
        if missing_prices:
            raise ValueError(f"No price data available for tickers: {', '.join(missing_prices)}")
        
        # Calculate portfolio metrics using vectorized operations
        df['current_value'] = df['current_shares'] * df['price']
        total_portfolio_value = df['current_value'].sum()
        print(f'Total Portfolio Market Value: ${total_portfolio_value:,.2f}')
        
        if total_portfolio_value <= 0:
            raise ValueError("Portfolio has no market value")
        
        df['current_weight'] = df['current_value'] / total_portfolio_value
        df['target_value'] = df['target_weight'] * total_portfolio_value
        df['target_shares'] = df['target_value'] / df['price']
        df['shares_diff'] = df['target_shares'] - df['current_shares']
        
        # Apply lot-based trading logic
        def calculate_lot_shares(shares_diff, lot_size):
            abs_diff = abs(shares_diff)
            if abs_diff < lot_size:
                return 0.0  # No trade if below lot size
            else:
                # Round down to nearest multiple of lot_size
                num_lots = int(abs_diff // lot_size)
                return num_lots * lot_size
        
        df['lot_shares'] = df['shares_diff'].apply(lambda x: calculate_lot_shares(x, validated_lot_size))
        df['trade_value'] = (df['lot_shares'] * df['price']).round(2)

        print('Calculated Rebalancing DataFrame:')
        print(df)
        
        
        # Determine actions based on lot-based trading
        def determine_action(shares_diff, lot_shares):
            if lot_shares == 0.0:
                return "HOLD"  # Below minimum lot size
            elif shares_diff > 0:
                return "BUY"
            else:
                return "SELL"
        
        df['action'] = df.apply(lambda row: determine_action(row['shares_diff'], row['lot_shares']), axis=1)
        
        # Calculate actual final portfolio after lot-based trading
        def calculate_final_shares(row):
            if row['action'] == 'BUY':
                return row['current_shares'] + row['lot_shares']
            elif row['action'] == 'SELL':
                return row['current_shares'] - row['lot_shares']
            else:  # HOLD
                return row['current_shares']
        
        df['final_shares'] = df.apply(calculate_final_shares, axis=1)
        df['final_value'] = df['final_shares'] * df['price']
        df['final_weight'] = df['final_value'] / total_portfolio_value
        
        print('Final Rebalancing Actions DataFrame:')
        print(df)
        
        # Filter for current holdings (non-zero positions)
        current_df = df[df['current_shares'] > 0].copy()
        current_holdings = [
            create_portfolio_holding(
                ticker=row['ticker'],
                shares=row['current_shares'],
                price=row['price'],
                market_value=row['current_value'],
                weight_percent=row['current_weight'] * 100
            ) for _, row in current_df.iterrows()
        ]
        
        # Build rebalancing actions
        rebalancing_actions = [
            create_rebalancing_action(
                ticker=row['ticker'],
                action=row['action'],
                current_weight_percent=row['current_weight'] * 100,
                target_weight_percent=row['target_weight'] * 100,
                shares=row['lot_shares'] if row['action'] != "HOLD" else 0.0,
                amount=row['trade_value'] if row['action'] != "HOLD" else 0.0
            ) for _, row in df.iterrows()
        ]
        
        # Filter for final holdings (positions with shares after lot-based trading)
        final_df = df[df['final_shares'] > 0].copy()
        target_holdings = [
            create_portfolio_holding(
                ticker=row['ticker'],
                shares=row['final_shares'],
                price=row['price'],
                market_value=row['final_value'],
                weight_percent=row['final_weight'] * 100
            ) for _, row in final_df.iterrows()
        ]
        
        # Calculate summary metrics
        total_trades_value = df[df['action'] != "HOLD"]['trade_value'].sum()
        number_of_actions = len(df[df['action'] != "HOLD"])
        
        return create_rebalancing_result(
            target_index=validated_index_name,
            current_portfolio_value=total_portfolio_value,
            lot_size=validated_lot_size,
            current_holdings=current_holdings,
            rebalancing_actions=rebalancing_actions,
            target_holdings=target_holdings,
            total_trading_value=total_trades_value,
            number_of_actions=number_of_actions
        )
        
    except Exception as e:
        return create_rebalancing_result(
            target_index="ERROR",
            current_portfolio_value=0.0,
            lot_size=validated_lot_size if 'validated_lot_size' in locals() else 1.0,
            current_holdings=[],
            rebalancing_actions=[],
            target_holdings=[],
            total_trading_value=0.0,
            number_of_actions=0
        )


if __name__ == "__main__":
    # Quick test of the tools
    print("🧪 Testing Portfolio Tools\n")
    
    # Test 1: Get index weights
    print("1. Testing get_index_weights:")
    result1 = get_index_weights.invoke({"index_name": "index1"})
    print(result1)
    print()
    
    # Test 2: Get prices
    print("2. Testing get_prices:")
    result2 = get_prices.invoke({"ticker": "AAPL"})
    print(result2)
    result3 = get_prices.invoke({"ticker": "NVDA"})
    print(result3)
    print()
    
    # Test 3: Rebalance portfolio
    print("3. Testing rebalance_portfolio:")
    test_holdings = {"AAPL": 100, "GOOGL": 20, "MSFT": 50}
    test_index_name = "index1"
    test_lot_size = 10.0
    result4 = rebalance_portfolio.invoke({
        "holdings": test_holdings,
        "index_name": test_index_name,
        "lot_size": test_lot_size
    })
    print(f"Result type: {type(result4)}")
    if isinstance(result4, dict):
        print(format_rebalancing_summary(result4))
        print(f"\n📊 Structured Result:")
        print(f"Target Index: {result4['target_index']}")
        print(f"Portfolio Value: ${result4['current_portfolio_value']:,.2f}")
        print(f"Trading Actions: {result4['number_of_actions']}")
        print(f"Trading Value: ${result4['total_trading_value']:,.2f}")
    else:
        print(result4)