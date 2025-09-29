#!/usr/bin/env python3
"""
Tax Loss Harvesting Tools for Portfolio Management
This module provides tools for tax loss harvesting analysis with FIFO and index-optimized strategies.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, date
from dataclasses import dataclass
from langchain_core.tools import tool
from finance.tools.portfolio_data import INDEX_WEIGHTS, STOCK_PRICES


@dataclass
class TaxLot:
    """Data class representing a tax lot for a security."""
    ticker: str
    shares: float
    cost_basis: float  # Cost per share
    purchase_date: date
    current_price: float = 0.0
    
    @property
    def total_cost(self) -> float:
        """Total cost basis of the lot."""
        return self.shares * self.cost_basis
    
    @property
    def current_value(self) -> float:
        """Current market value of the lot."""
        return self.shares * self.current_price
    
    @property
    def unrealized_gain_loss(self) -> float:
        """Unrealized gain/loss for the lot."""
        return self.current_value - self.total_cost
    
    @property
    def gain_loss_per_share(self) -> float:
        """Gain/loss per share."""
        return self.current_price - self.cost_basis
    
    def is_loss(self) -> bool:
        """Check if this lot has an unrealized loss."""
        return self.unrealized_gain_loss < 0


@dataclass
class TLHAction:
    """Data class representing a tax loss harvesting action."""
    ticker: str
    action: str  # "SELL" or "HOLD"
    lot_index: int
    shares_to_sell: float
    cost_basis: float
    current_price: float
    realized_loss: float
    purchase_date: date
    
    @property
    def proceeds(self) -> float:
        """Sale proceeds from the action."""
        return self.shares_to_sell * self.current_price
    
    @property
    def total_cost_basis(self) -> float:
        """Total cost basis of shares to sell."""
        return self.shares_to_sell * self.cost_basis


@dataclass
class TLHResult:
    """Data class representing tax loss harvesting analysis results."""
    total_portfolio_value: float
    lot_size: float
    max_sell_percentage: float
    index_name: Optional[str]
    actions: List[TLHAction]
    total_realized_losses: float
    total_proceeds: float
    total_shares_sold: float
    portfolio_impact_percentage: float
    tracking_error_impact: Optional[float] = None
    
    @property
    def number_of_sales(self) -> int:
        """Number of sell actions."""
        return len([a for a in self.actions if a.action == "SELL"])


def validate_tax_lots(tax_lots: List[Dict[str, Any]]) -> List[TaxLot]:
    """Validate and convert tax lots from dictionary format to TaxLot objects."""
    if not tax_lots:
        raise ValueError("Tax lots list cannot be empty")
    
    validated_lots = []
    for i, lot_data in enumerate(tax_lots):
        try:
            # Required fields
            ticker = str(lot_data.get("ticker", "")).upper().strip()
            if not ticker:
                raise ValueError(f"Lot {i}: ticker is required")
            
            shares = float(lot_data.get("shares", 0))
            if shares <= 0:
                raise ValueError(f"Lot {i}: shares must be positive")
            
            cost_basis = float(lot_data.get("cost_basis", 0))
            if cost_basis <= 0:
                raise ValueError(f"Lot {i}: cost_basis must be positive")
            
            # Handle purchase_date - can be string or date object
            purchase_date_raw = lot_data.get("purchase_date")
            if isinstance(purchase_date_raw, str):
                try:
                    purchase_date = datetime.strptime(purchase_date_raw, "%Y-%m-%d").date()
                except ValueError:
                    raise ValueError(f"Lot {i}: purchase_date must be in YYYY-MM-DD format")
            elif isinstance(purchase_date_raw, date):
                purchase_date = purchase_date_raw
            else:
                raise ValueError(f"Lot {i}: purchase_date is required")
            
            # Get current price from STOCK_PRICES
            if ticker not in STOCK_PRICES:
                raise ValueError(f"Lot {i}: No price data available for ticker {ticker}")
            
            current_price = STOCK_PRICES[ticker]
            
            validated_lots.append(TaxLot(
                ticker=ticker,
                shares=shares,
                cost_basis=cost_basis,
                purchase_date=purchase_date,
                current_price=current_price
            ))
            
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"Invalid tax lot at index {i}: {str(e)}")
    
    return validated_lots




def _tax_loss_harvest_fifo(tax_lots: List[TaxLot],
                          lot_size: float = 1.0,
                          max_sell_percentage: float = 100.0) -> TLHResult:
    """
    Perform tax loss harvesting using FIFO (First In, First Out) strategy.
    
    Args:
        tax_lots: List of TaxLot objects
        lot_size: Minimum trading lot size in shares
        max_sell_percentage: Maximum percentage of each ticker's total shares to sell
        
    Returns:
        TLHResult object with harvesting analysis
    """
    if not tax_lots:
        raise ValueError("Tax lots list cannot be empty")
    
    if lot_size <= 0:
        raise ValueError("lot_size must be greater than 0")
    
    if not (0 < max_sell_percentage <= 100):
        raise ValueError("max_sell_percentage must be between 0 and 100")
    
    # Calculate total portfolio value and per-ticker share totals
    total_portfolio_value = sum(lot.current_value for lot in tax_lots)
    ticker_share_totals = {}
    for lot in tax_lots:
        if lot.ticker not in ticker_share_totals:
            ticker_share_totals[lot.ticker] = 0.0
        ticker_share_totals[lot.ticker] += lot.shares
    
    # Calculate max sellable shares per ticker
    ticker_max_sell_shares = {}
    for ticker, total_shares in ticker_share_totals.items():
        ticker_max_sell_shares[ticker] = total_shares * (max_sell_percentage / 100.0)
    
    # Sort lots by purchase date (FIFO)
    sorted_lots = sorted(tax_lots, key=lambda x: x.purchase_date)
    
    actions = []
    total_proceeds = 0.0
    ticker_shares_sold = {}  # Track shares sold per ticker
    
    # Initialize ticker tracking
    for ticker in ticker_share_totals:
        ticker_shares_sold[ticker] = 0.0
    
    for lot_index, lot in enumerate(sorted_lots):
        # Check if we've reached the max sell limit for this ticker
        remaining_sell_capacity = ticker_max_sell_shares[lot.ticker] - ticker_shares_sold[lot.ticker]
        
        if remaining_sell_capacity <= 0:
            # Already hit max sell limit for this ticker
            actions.append(TLHAction(
                ticker=lot.ticker,
                action="HOLD",
                lot_index=lot_index,
                shares_to_sell=0.0,
                cost_basis=lot.cost_basis,
                current_price=lot.current_price,
                realized_loss=0.0,
                purchase_date=lot.purchase_date
            ))
            continue
        
        if lot.is_loss():
            # Calculate maximum shares we can sell within ticker limit
            max_shares_available = min(lot.shares, remaining_sell_capacity)
            
            # Apply lot size constraints
            if max_shares_available >= lot_size:
                shares_to_sell = int(max_shares_available // lot_size) * lot_size
                
                if shares_to_sell > 0:
                    realized_loss = shares_to_sell * lot.gain_loss_per_share
                    proceeds = shares_to_sell * lot.current_price
                    
                    actions.append(TLHAction(
                        ticker=lot.ticker,
                        action="SELL",
                        lot_index=lot_index,
                        shares_to_sell=shares_to_sell,
                        cost_basis=lot.cost_basis,
                        current_price=lot.current_price,
                        realized_loss=realized_loss,
                        purchase_date=lot.purchase_date
                    ))
                    
                    total_proceeds += proceeds
                    ticker_shares_sold[lot.ticker] += shares_to_sell
                else:
                    # Below lot size threshold
                    actions.append(TLHAction(
                        ticker=lot.ticker,
                        action="HOLD",
                        lot_index=lot_index,
                        shares_to_sell=0.0,
                        cost_basis=lot.cost_basis,
                        current_price=lot.current_price,
                        realized_loss=0.0,
                        purchase_date=lot.purchase_date
                    ))
            else:
                # Below lot size threshold
                actions.append(TLHAction(
                    ticker=lot.ticker,
                    action="HOLD",
                    lot_index=lot_index,
                    shares_to_sell=0.0,
                    cost_basis=lot.cost_basis,
                    current_price=lot.current_price,
                    realized_loss=0.0,
                    purchase_date=lot.purchase_date
                ))
        else:
            # No loss, hold the lot
            actions.append(TLHAction(
                ticker=lot.ticker,
                action="HOLD",
                lot_index=lot_index,
                shares_to_sell=0.0,
                cost_basis=lot.cost_basis,
                current_price=lot.current_price,
                realized_loss=0.0,
                purchase_date=lot.purchase_date
            ))
    
    # Calculate summary metrics
    total_realized_losses = sum(action.realized_loss for action in actions if action.action == "SELL")
    total_shares_sold = sum(action.shares_to_sell for action in actions if action.action == "SELL")
    portfolio_impact_percentage = (total_proceeds / total_portfolio_value) * 100
    
    return TLHResult(
        total_portfolio_value=total_portfolio_value,
        lot_size=lot_size,
        max_sell_percentage=max_sell_percentage,
        index_name=None,
        actions=actions,
        total_realized_losses=total_realized_losses,
        total_proceeds=total_proceeds,
        total_shares_sold=total_shares_sold,
        portfolio_impact_percentage=portfolio_impact_percentage
    )


def _tax_loss_harvest_index_optimized(tax_lots: List[TaxLot],
                                   index_name: str,
                                   lot_size: float = 1.0,
                                   allocation_tolerance: float = 5.0,
                                   verbose: bool = False) -> TLHResult:
    """
    Perform tax loss harvesting with allocation tolerance constraints.
    
    Args:
        tax_lots: List of TaxLot objects
        index_name: Target index for allocation tracking
        lot_size: Minimum trading lot size in shares
        allocation_tolerance: Maximum allowed percentage deviation from target weights per ticker
        verbose: Enable detailed logging for debugging
        
    Returns:
        TLHResult object with harvesting analysis
    """
    if not tax_lots:
        raise ValueError("Tax lots list cannot be empty")
    
    if lot_size <= 0:
        raise ValueError("lot_size must be greater than 0")
    
    if not (0 <= allocation_tolerance <= 100):
        raise ValueError("allocation_tolerance must be between 0 and 100")
    
    if index_name.lower() not in INDEX_WEIGHTS:
        raise ValueError(f"Invalid index name: {index_name}")
    
    # Get target index weights
    indexWeights = INDEX_WEIGHTS[index_name.lower()]
    
    # Calculate total portfolio value (use original value throughout)
    total_portfolio_value = sum(lot.current_value for lot in tax_lots)
    
    # Phase 1: Initialize Weight Tracking
    originalWeights = {}
    for lot in tax_lots:
        ticker_weight = lot.current_value / total_portfolio_value
        if lot.ticker in originalWeights:
            originalWeights[lot.ticker] += ticker_weight
        else:
            originalWeights[lot.ticker] = ticker_weight
    
    # Initialize updatedWeights with originalWeights
    updatedWeights = originalWeights.copy()
    
    if verbose:
        print(f"\n📊 Simplified Index-Optimized TLH")
        print(f"=" * 50)
        print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")
        print(f"Allocation Tolerance: {allocation_tolerance}%")
        print(f"Lot Size: {lot_size} shares")
        print(f"Target Index: {index_name.upper()}")
        print(f"\n📈 Initial Weights:")
        for ticker, weight in originalWeights.items():
            target = indexWeights.get(ticker, 0.0)
            print(f"   {ticker}: Current {weight:.1%}, Target {target:.1%}")
    
    # Phase 2: Process Each Lot Sequentially
    actions = []
    total_proceeds = 0.0
    total_shares_sold = 0.0
    
    if verbose:
        print(f"\n🔍 Processing Lots:")
    
    for lot_index, lot in enumerate(tax_lots):
        if verbose:
            print(f"\n   Lot {lot_index}: {lot.ticker}")
            print(f"     Shares: {lot.shares}, Cost: ${lot.cost_basis:.2f}, Current: ${lot.current_price:.2f}")
            print(f"     P&L: ${lot.unrealized_gain_loss:,.2f}")
        
        # Phase 3: Sale Decision Logic
        if lot.is_loss():
            if verbose:
                print(f"     🔴 LOSS LOT - Evaluating for sale")
            
            # Calculate lot's weight impact
            lot_weight = lot.current_value / total_portfolio_value
            target_weight = indexWeights.get(lot.ticker, 0.0)  # Missing tickers = 0% target
            current_weight = updatedWeights[lot.ticker]
            
            # Check if full sale fits within tolerance
            newPotentialWeight = current_weight - lot_weight
            deviation_after_sale = abs(newPotentialWeight - target_weight) * 100
            
            if verbose:
                print(f"     Current Weight: {current_weight:.1%}, Target: {target_weight:.1%}")
                print(f"     Lot Weight Impact: {lot_weight:.1%}")
                print(f"     New Potential Weight: {newPotentialWeight:.1%}")
                print(f"     Deviation After Sale: {deviation_after_sale:.1f}%")
            
            if deviation_after_sale <= allocation_tolerance:
                # Sell whole lot (with lot size adjustment)
                shares_to_sell = int(lot.shares // lot_size) * lot_size
                
                if shares_to_sell >= lot_size:
                    realized_loss = shares_to_sell * lot.gain_loss_per_share
                    proceeds = shares_to_sell * lot.current_price
                    
                    # Update weights
                    sold_weight = (shares_to_sell * lot.current_price) / total_portfolio_value
                    updatedWeights[lot.ticker] = max(0.0, updatedWeights[lot.ticker] - sold_weight)
                    
                    if verbose:
                        print(f"     ✅ FULL SALE: {shares_to_sell} shares for ${proceeds:,.2f} (${realized_loss:,.2f} loss)")
                        print(f"     Updated Weight: {updatedWeights[lot.ticker]:.1%}")
                    
                    actions.append(TLHAction(
                        ticker=lot.ticker,
                        action="SELL",
                        lot_index=lot_index,
                        shares_to_sell=shares_to_sell,
                        cost_basis=lot.cost_basis,
                        current_price=lot.current_price,
                        realized_loss=realized_loss,
                        purchase_date=lot.purchase_date
                    ))
                    
                    total_proceeds += proceeds
                    total_shares_sold += shares_to_sell
                else:
                    if verbose:
                        print(f"     ❌ BELOW LOT SIZE: {shares_to_sell} < {lot_size}")
                    
                    actions.append(TLHAction(
                        ticker=lot.ticker,
                        action="HOLD",
                        lot_index=lot_index,
                        shares_to_sell=0.0,
                        cost_basis=lot.cost_basis,
                        current_price=lot.current_price,
                        realized_loss=0.0,
                        purchase_date=lot.purchase_date
                    ))
            else:
                # Calculate partial sale using algebraic approach
                if verbose:
                    print(f"     🔄 ATTEMPTING PARTIAL SALE")
                
                # Calculate minimum allowed weight and maximum weight reduction
                # Example: Current=40%, Target=20%, Tolerance=5% -> Min=15%, MaxReduction=25%
                tolerance_decimal = allocation_tolerance / 100.0
                min_allowed_weight = target_weight - tolerance_decimal
                max_weight_reduction = max(0.0, current_weight - min_allowed_weight)
                
                if max_weight_reduction > 0:
                    max_sellable_value = max_weight_reduction * total_portfolio_value
                    max_shares_algebraic = max_sellable_value / lot.current_price
                    max_shares_lot_adjusted = int(max_shares_algebraic // lot_size) * lot_size
                    
                    if max_shares_lot_adjusted >= lot_size and max_shares_lot_adjusted <= lot.shares:
                        realized_loss = max_shares_lot_adjusted * lot.gain_loss_per_share
                        proceeds = max_shares_lot_adjusted * lot.current_price
                        
                        # Update weights
                        sold_weight = proceeds / total_portfolio_value
                        updatedWeights[lot.ticker] = max(0.0, updatedWeights[lot.ticker] - sold_weight)
                        
                        if verbose:
                            print(f"     ✅ PARTIAL SALE: {max_shares_lot_adjusted} shares for ${proceeds:,.2f} (${realized_loss:,.2f} loss)")
                            print(f"     Updated Weight: {updatedWeights[lot.ticker]:.1%}")
                        
                        actions.append(TLHAction(
                            ticker=lot.ticker,
                            action="SELL",
                            lot_index=lot_index,
                            shares_to_sell=max_shares_lot_adjusted,
                            cost_basis=lot.cost_basis,
                            current_price=lot.current_price,
                            realized_loss=realized_loss,
                            purchase_date=lot.purchase_date
                        ))
                        
                        total_proceeds += proceeds
                        total_shares_sold += max_shares_lot_adjusted
                    else:
                        if verbose:
                            print(f"     ❌ PARTIAL SALE NOT VIABLE: Max shares {max_shares_lot_adjusted} < lot size {lot_size}")
                        
                        actions.append(TLHAction(
                            ticker=lot.ticker,
                            action="HOLD",
                            lot_index=lot_index,
                            shares_to_sell=0.0,
                            cost_basis=lot.cost_basis,
                            current_price=lot.current_price,
                            realized_loss=0.0,
                            purchase_date=lot.purchase_date
                        ))
                else:
                    if verbose:
                        print(f"     ❌ ALREADY WITHIN TOLERANCE: No sale needed")
                    
                    actions.append(TLHAction(
                        ticker=lot.ticker,
                        action="HOLD",
                        lot_index=lot_index,
                        shares_to_sell=0.0,
                        cost_basis=lot.cost_basis,
                        current_price=lot.current_price,
                        realized_loss=0.0,
                        purchase_date=lot.purchase_date
                    ))
        else:
            if verbose:
                print(f"     🟢 GAIN LOT - Hold")
            
            actions.append(TLHAction(
                ticker=lot.ticker,
                action="HOLD",
                lot_index=lot_index,
                shares_to_sell=0.0,
                cost_basis=lot.cost_basis,
                current_price=lot.current_price,
                realized_loss=0.0,
                purchase_date=lot.purchase_date
            ))
    
    # Phase 4: Calculate Results
    total_realized_losses = sum(action.realized_loss for action in actions if action.action == "SELL")
    portfolio_impact_percentage = (total_proceeds / total_portfolio_value) * 100
    
    # Calculate maximum final deviation
    max_final_deviation = 0.0
    for ticker in set(list(updatedWeights.keys()) + list(indexWeights.keys())):
        final_weight = updatedWeights.get(ticker, 0.0)
        target_weight = indexWeights.get(ticker, 0.0)
        deviation = abs(final_weight - target_weight) * 100
        max_final_deviation = max(max_final_deviation, deviation)
    
    if verbose:
        print(f"\n📈 Final Weights:")
        for ticker in set(list(updatedWeights.keys()) + list(indexWeights.keys())):
            final_weight = updatedWeights.get(ticker, 0.0)
            target_weight = indexWeights.get(ticker, 0.0)
            print(f"   {ticker}: Current {final_weight:.1%}, Target {target_weight:.1%}")
        print(f"   Sales Proceeds: ${total_proceeds:,.2f}")
        
        print(f"\n📋 Final Results:")
        print(f"   Total Realized Losses: ${total_realized_losses:,.2f}")
        print(f"   Total Proceeds: ${total_proceeds:,.2f}")
        print(f"   Shares Sold: {total_shares_sold:,.0f}")
        print(f"   Portfolio Impact: {portfolio_impact_percentage:.1f}%")
        print(f"   Max Final Deviation: {max_final_deviation:.1f}%")
    
    return TLHResult(
        total_portfolio_value=total_portfolio_value,
        lot_size=lot_size,
        max_sell_percentage=allocation_tolerance,
        index_name=index_name.lower(),
        actions=actions,
        total_realized_losses=total_realized_losses,
        total_proceeds=total_proceeds,
        total_shares_sold=total_shares_sold,
        portfolio_impact_percentage=portfolio_impact_percentage,
        tracking_error_impact=max_final_deviation
    )


@tool
def tax_loss_harvest(tax_lots: List[Dict[str, Any]],
                    index_name: Optional[str] = None,
                    lot_size: float = 1.0,
                    max_sell_percentage: Optional[float] = None,
                    allocation_tolerance: Optional[float] = None,
                    verbose: bool = False) -> Dict[str, Any]:
    """
    Analyze tax lot positions for tax loss harvesting opportunities using FIFO or index-optimized strategies.
    
    This function performs comprehensive tax loss harvesting analysis on a portfolio of tax lots,
    automatically selecting between FIFO (First In, First Out) and index-optimized strategies based
    on whether an index target is specified. It identifies loss positions and recommends sales while
    respecting lot size constraints and portfolio allocation limits.
    
    Args:
        tax_lots (List[Dict[str, Any]]): List of tax lot dictionaries representing individual security positions.
            Each tax lot dictionary must contain:
            - ticker (str): Stock ticker symbol (e.g., "AAPL", "GOOGL")
            - shares (float): Number of shares in the tax lot (must be > 0)
            - cost_basis (float): Original cost per share when purchased (must be > 0)
            - purchase_date (str|date): Purchase date in "YYYY-MM-DD" format or date object
            
            Example: [
                {"ticker": "AAPL", "shares": 100.0, "cost_basis": 150.0, "purchase_date": "2023-01-15"},
                {"ticker": "TSLA", "shares": 50.0, "cost_basis": 300.0, "purchase_date": "2023-02-10"}
            ]
            
        index_name (Optional[str]): Target index name for index-optimized strategy.
            - If provided: Uses index-optimized strategy with allocation tolerance constraints
            - If None: Uses FIFO strategy with max sell percentage constraints
            - Valid values: "index1", "index2", "index3" (case-insensitive)
            - "index1": 5-stock tech portfolio (AAPL, GOOGL, MSFT, AMZN, TSLA)
            - "index2": 10-stock diversified tech portfolio
            - "index3": 5-stock balanced portfolio (ABC, XYZ, PQR, LMN, EFG)
            
        lot_size (float): Minimum trading lot size in shares per transaction.
            - Must be > 0.0
            - Default: 1.0 (no lot size constraint)
            - Example: 10.0 means sales must be in multiples of 10 shares
            - Sales below lot_size threshold will result in "HOLD" action
            
        max_sell_percentage (Optional[float]): Maximum percentage of total portfolio value to sell (FIFO strategy only).
            - Used only when index_name is None (FIFO strategy)
            - Must be between 0 and 100
            - Default: 100.0 (no limit)
            - Example: 50.0 means maximum 50% of portfolio value can be sold
            
        allocation_tolerance (Optional[float]): Maximum allowed percentage point deviation from target weights (index-optimized strategy only).
            - Used only when index_name is provided (index-optimized strategy)
            - Must be between 0 and 100
            - Default: 5.0 (5 percentage point tolerance)
            - Example: 5.0 means position can deviate ±5% from target weight
            - Controls how aggressively lots are sold while maintaining index tracking
            
        verbose (bool): Enable detailed logging output for debugging and analysis.
            - Default: False
            - When True: Prints detailed weight calculations, sale decisions, and final allocation analysis
            
    Returns:
        Dict[str, Any]: Comprehensive tax loss harvesting analysis with the following structure:
        
        {
            "strategy": str,                         # "fifo" or "index_optimized"
            "total_portfolio_value": float,          # Total market value of all tax lots
            "lot_size": float,                      # Lot size used for calculations
            "max_sell_percentage": float,           # Max sell limit used (FIFO) or allocation_tolerance (index-optimized)
            "index_name": Optional[str],            # Target index name (None for FIFO)
            "total_realized_losses": float,         # Total realized losses from all sales (negative number)
            "total_proceeds": float,                # Total proceeds from all sales
            "total_shares_sold": float,             # Total number of shares sold across all lots
            "number_of_sales": int,                 # Count of actual sale transactions
            "portfolio_impact_percentage": float,   # Percentage of portfolio affected by sales
            "tracking_error_impact": Optional[float], # Final tracking error vs target index (index-optimized only)
            
            "actions": List[Dict[str, Any]]         # Detailed action recommendations for each tax lot
            # Each action dictionary contains:
            # {
            #     "ticker": str,                    # Stock symbol
            #     "action": str,                    # "SELL" or "HOLD"
            #     "lot_index": int,                 # Index of lot in original tax_lots list
            #     "shares_to_sell": float,          # Number of shares to sell (0 if HOLD)
            #     "cost_basis": float,              # Original cost per share
            #     "current_price": float,           # Current market price per share
            #     "realized_loss": float,           # Realized loss from sale (0 if HOLD)
            #     "purchase_date": str,             # Purchase date in ISO format
            #     "proceeds": float,                # Sale proceeds (shares_to_sell * current_price)
            #     "total_cost_basis": float         # Total cost basis of shares sold
            # }
        }
        
        Error Response (if invalid inputs):
        {
            "error": str,                           # Error description
            "strategy": "error",                    # Indicates error state
            # ... other fields set to default values
        }
    
    Strategy Selection Logic:
        - If index_name is provided → Index-Optimized Strategy
            * Prioritizes maintaining target index allocation within tolerance
            * Uses allocation_tolerance to control maximum deviation per ticker
            * Attempts partial sales to stay within tolerance bounds
            * Suitable for index-tracking portfolios
            
        - If index_name is None → FIFO Strategy
            * Sells loss lots in chronological order (oldest first)
            * Uses max_sell_percentage to limit total portfolio impact
            * Simpler strategy focused purely on loss harvesting
            * Suitable for non-index portfolios
    
    Example Usage:
        >>> # FIFO Strategy Example
        >>> tax_lots = [
        ...     {"ticker": "AAPL", "shares": 100, "cost_basis": 200.0, "purchase_date": "2023-01-15"},
        ...     {"ticker": "TSLA", "shares": 50, "cost_basis": 300.0, "purchase_date": "2023-02-10"}
        ... ]
        >>> result = tax_loss_harvest(tax_lots, lot_size=10.0, max_sell_percentage=50.0)
        >>> print(f"Realized losses: ${result['total_realized_losses']:,.2f}")
        >>> print(f"Sales executed: {result['number_of_sales']}")
        
        >>> # Index-Optimized Strategy Example
        >>> result = tax_loss_harvest(tax_lots, index_name="index1", allocation_tolerance=3.0)
        >>> print(f"Strategy: {result['strategy']}")
        >>> print(f"Tracking error: {result['tracking_error_impact']:.2f}%")
        >>> for action in result['actions']:
        ...     if action['action'] == 'SELL':
        ...         print(f"Sell {action['shares_to_sell']} shares of {action['ticker']}")
    
    Raises:
        ValueError: If invalid input parameters are provided:
            - Empty tax_lots list
            - Invalid ticker symbols or missing price data
            - Invalid lot_size (must be > 0)
            - Invalid percentage values (must be 0-100)
            - Invalid index_name (must be valid index key)
            - Missing or invalid required fields in tax lot dictionaries
            
    Notes:
        - Current stock prices are sourced from hardcoded STOCK_PRICES dictionary in portfolio_tools
        - Index target weights are defined in INDEX_WEIGHTS dictionary in portfolio_tools
        - Tax lots with unrealized gains are automatically held (no sales)
        - Partial lot sales respect lot_size constraints (rounded down to nearest multiple)
        - Function returns structured data suitable for programmatic processing
        - All monetary values are in USD
        - Purchase dates are converted to ISO format in output
        - Strategy automatically adapts based on presence of index_name parameter
        - Verbose output provides detailed decision-making transparency for analysis
    """
    try:
        # Validate and convert tax lots
        validated_lots = validate_tax_lots(tax_lots)
        
        # Select strategy and validate parameters
        if index_name:
            # Index-optimized strategy
            if allocation_tolerance is None:
                allocation_tolerance = 5.0  # Default 5% tolerance
            result = _tax_loss_harvest_index_optimized(
                validated_lots, index_name, lot_size, allocation_tolerance, verbose
            )
        else:
            # FIFO strategy
            if max_sell_percentage is None:
                max_sell_percentage = 100.0  # Default 100% (no limit)
            result = _tax_loss_harvest_fifo(
                validated_lots, lot_size, max_sell_percentage
            )
        
        # Convert result to dictionary format
        return {
            "strategy": "index_optimized" if index_name else "fifo",
            "total_portfolio_value": result.total_portfolio_value,
            "lot_size": result.lot_size,
            "max_sell_percentage": result.max_sell_percentage,
            "index_name": result.index_name,
            "total_realized_losses": result.total_realized_losses,
            "total_proceeds": result.total_proceeds,
            "total_shares_sold": result.total_shares_sold,
            "number_of_sales": result.number_of_sales,
            "portfolio_impact_percentage": result.portfolio_impact_percentage,
            "tracking_error_impact": result.tracking_error_impact,
            "actions": [
                {
                    "ticker": action.ticker,
                    "action": action.action,
                    "lot_index": action.lot_index,
                    "shares_to_sell": action.shares_to_sell,
                    "cost_basis": action.cost_basis,
                    "current_price": action.current_price,
                    "realized_loss": action.realized_loss,
                    "purchase_date": action.purchase_date.isoformat(),
                    "proceeds": action.proceeds,
                    "total_cost_basis": action.total_cost_basis
                }
                for action in result.actions
            ]
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "strategy": "error",
            "total_portfolio_value": 0.0,
            "lot_size": lot_size,
            "max_sell_percentage": max_sell_percentage,
            "index_name": index_name,
            "total_realized_losses": 0.0,
            "total_proceeds": 0.0,
            "total_shares_sold": 0.0,
            "number_of_sales": 0,
            "portfolio_impact_percentage": 0.0,
            "tracking_error_impact": None,
            "actions": []
        }


def format_tlh_summary(result: Dict[str, Any]) -> str:
    """Format the TLH result as a human-readable string."""
    if "error" in result:
        return f"❌ Tax Loss Harvesting Error: {result['error']}"
    
    output = f"💰 Tax Loss Harvesting Analysis\n"
    output += "=" * 50 + "\n"
    output += f"Strategy: {result['strategy'].upper()}\n"
    output += f"Portfolio Value: ${result['total_portfolio_value']:,.2f}\n"
    output += f"Lot Size: {result['lot_size']} shares\n"
    output += f"Max Sell Limit: {result['max_sell_percentage']}%\n"
    if result['index_name']:
        output += f"Target Index: {result['index_name'].upper()}\n"
    output += f"\n📊 Results Summary:\n"
    output += "-" * 30 + "\n"
    output += f"Total Realized Losses: ${result['total_realized_losses']:,.2f}\n"
    output += f"Total Proceeds: ${result['total_proceeds']:,.2f}\n"
    output += f"Shares Sold: {result['total_shares_sold']:,.0f}\n"
    output += f"Number of Sales: {result['number_of_sales']}\n"
    output += f"Portfolio Impact: {result['portfolio_impact_percentage']:.1f}%\n"
    
    if result['tracking_error_impact'] is not None:
        output += f"Tracking Error Impact: {result['tracking_error_impact']:.4f}\n"
    
    # Show individual actions
    output += f"\n📋 Tax Loss Harvesting Actions:\n"
    output += "-" * 50 + "\n"
    output += f"{'Ticker':<6} {'Action':<6} {'Shares':<8} {'Loss':<12} {'Date':<12}\n"
    output += "-" * 50 + "\n"
    
    for action in result['actions']:
        if action['action'] == 'SELL':
            output += f"{action['ticker']:<6} {action['action']:<6} {action['shares_to_sell']:>8.0f} "
            output += f"${action['realized_loss']:>10.2f} {action['purchase_date']:<12}\n"
    
    return output


if __name__ == "__main__":
    # Quick test of the TLH system
    print("🧪 Testing Tax Loss Harvesting System\n")
    
    # Test data with tax lots having different cost bases and purchase dates
    test_tax_lots = [
        {
            "ticker": "AAPL",
            "shares": 100.0,
            "cost_basis": 200.0,  # Loss: current $185 vs cost $200
            "purchase_date": "2023-01-15"
        },
        {
            "ticker": "AAPL", 
            "shares": 50.0,
            "cost_basis": 150.0,  # Gain: current $185 vs cost $150
            "purchase_date": "2023-06-20"
        },
        {
            "ticker": "GOOGL",
            "shares": 10.0,
            "cost_basis": 3000.0,  # Loss: current $2800 vs cost $3000
            "purchase_date": "2023-02-10"
        },
        {
            "ticker": "MSFT",
            "shares": 25.0,
            "cost_basis": 400.0,  # Loss: current $375 vs cost $400
            "purchase_date": "2023-03-05"
        }
    ]
    
    # Test 1: FIFO-based TLH
    print("1. Testing FIFO Tax Loss Harvesting:")
    fifo_result = tax_loss_harvest.invoke({
        "tax_lots": test_tax_lots,
        "lot_size": 10.0,
        "max_sell_percentage": 50.0
    })
    print(format_tlh_summary(fifo_result))
    print()
    
    # Test 2: Index-optimized TLH
    print("2. Testing Index-Optimized Tax Loss Harvesting:")
    index_result = tax_loss_harvest.invoke({
        "tax_lots": test_tax_lots,
        "index_name": "index1",
        "lot_size": 10.0,
        "max_sell_percentage": 50.0
    })
    print(format_tlh_summary(index_result))