#!/usr/bin/env python3
"""
Tax Loss Harvesting Tools for Portfolio Management
This module provides tools for tax loss harvesting analysis with FIFO and index-optimized strategies.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, date
from dataclasses import dataclass
from portfolio_tools import INDEX_WEIGHTS, STOCK_PRICES


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


def calculate_tracking_error_impact(current_weights: Dict[str, float], 
                                   new_weights: Dict[str, float], 
                                   target_weights: Dict[str, float]) -> float:
    """Calculate the tracking error impact of portfolio changes."""
    current_tracking_error = sum(
        (current_weights.get(ticker, 0) - target_weights.get(ticker, 0)) ** 2 
        for ticker in set(list(current_weights.keys()) + list(target_weights.keys()))
    ) ** 0.5
    
    new_tracking_error = sum(
        (new_weights.get(ticker, 0) - target_weights.get(ticker, 0)) ** 2 
        for ticker in set(list(new_weights.keys()) + list(target_weights.keys()))
    ) ** 0.5
    
    return new_tracking_error - current_tracking_error


def _tax_loss_harvest_fifo(tax_lots: List[TaxLot], 
                         lot_size: float = 1.0,
                         max_sell_percentage: float = 100.0) -> TLHResult:
    """
    Perform tax loss harvesting using FIFO (First In, First Out) strategy.
    
    Args:
        tax_lots: List of TaxLot objects
        lot_size: Minimum trading lot size in shares
        max_sell_percentage: Maximum percentage of portfolio to sell
        
    Returns:
        TLHResult object with harvesting analysis
    """
    if not tax_lots:
        raise ValueError("Tax lots list cannot be empty")
    
    if lot_size <= 0:
        raise ValueError("lot_size must be greater than 0")
    
    if not (0 < max_sell_percentage <= 100):
        raise ValueError("max_sell_percentage must be between 0 and 100")
    
    # Calculate total portfolio value
    total_portfolio_value = sum(lot.current_value for lot in tax_lots)
    max_sell_value = total_portfolio_value * (max_sell_percentage / 100.0)
    
    # Sort lots by purchase date (FIFO)
    sorted_lots = sorted(tax_lots, key=lambda x: x.purchase_date)
    
    actions = []
    total_proceeds = 0.0
    current_sell_value = 0.0
    
    for lot_index, lot in enumerate(sorted_lots):
        if current_sell_value >= max_sell_value:
            # Add HOLD action for remaining lots
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
            # Calculate maximum shares we can sell within constraints
            remaining_sell_capacity = max_sell_value - current_sell_value
            max_shares_by_value = remaining_sell_capacity / lot.current_price
            max_shares_available = min(lot.shares, max_shares_by_value)
            
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
                    current_sell_value += proceeds
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
    portfolio_impact_percentage = (current_sell_value / total_portfolio_value) * 100
    
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
                                   max_sell_percentage: float = 100.0) -> TLHResult:
    """
    Perform tax loss harvesting with index tracking error minimization.
    
    Args:
        tax_lots: List of TaxLot objects
        index_name: Target index for tracking error minimization
        lot_size: Minimum trading lot size in shares
        max_sell_percentage: Maximum percentage of portfolio to sell
        
    Returns:
        TLHResult object with harvesting analysis
    """
    if not tax_lots:
        raise ValueError("Tax lots list cannot be empty")
    
    if lot_size <= 0:
        raise ValueError("lot_size must be greater than 0")
    
    if not (0 < max_sell_percentage <= 100):
        raise ValueError("max_sell_percentage must be between 0 and 100")
    
    if index_name.lower() not in INDEX_WEIGHTS:
        raise ValueError(f"Invalid index name: {index_name}")
    
    target_weights = INDEX_WEIGHTS[index_name.lower()]
    
    # Calculate total portfolio value and current weights
    total_portfolio_value = sum(lot.current_value for lot in tax_lots)
    max_sell_value = total_portfolio_value * (max_sell_percentage / 100.0)
    
    # Calculate current portfolio weights by ticker
    current_weights = {}
    for lot in tax_lots:
        if lot.ticker in current_weights:
            current_weights[lot.ticker] += lot.current_value / total_portfolio_value
        else:
            current_weights[lot.ticker] = lot.current_value / total_portfolio_value
    
    # Identify loss lots and calculate tracking error impact for each potential sale
    loss_lots_with_impact = []
    
    for lot_index, lot in enumerate(tax_lots):
        if lot.is_loss():
            # Calculate potential shares to sell considering lot size
            max_shares_available = lot.shares
            if max_shares_available >= lot_size:
                shares_to_sell = int(max_shares_available // lot_size) * lot_size
                
                # Calculate new weights if we sell these shares
                proceeds = shares_to_sell * lot.current_price
                new_weights = current_weights.copy()
                new_weight = (current_weights.get(lot.ticker, 0) * total_portfolio_value - proceeds) / total_portfolio_value
                new_weights[lot.ticker] = max(0, new_weight)
                
                # Calculate tracking error impact
                tracking_impact = calculate_tracking_error_impact(current_weights, new_weights, target_weights)
                loss_per_tracking_unit = abs(shares_to_sell * lot.gain_loss_per_share) / (abs(tracking_impact) + 1e-8)
                
                loss_lots_with_impact.append({
                    'lot_index': lot_index,
                    'lot': lot,
                    'shares_to_sell': shares_to_sell,
                    'proceeds': proceeds,
                    'realized_loss': shares_to_sell * lot.gain_loss_per_share,
                    'tracking_impact': tracking_impact,
                    'loss_per_tracking_unit': loss_per_tracking_unit
                })
    
    # Sort by loss efficiency (minimize tracking error per unit of loss harvested)
    loss_lots_with_impact.sort(key=lambda x: -x['loss_per_tracking_unit'])
    
    # Select lots to sell within constraints
    actions = []
    selected_sales = []
    current_sell_value = 0.0
    
    # Select optimal sales with partial sale capability
    for lot_info in loss_lots_with_impact:
        if current_sell_value + lot_info['proceeds'] <= max_sell_value:
            # Full sale fits within limit
            selected_sales.append(lot_info)
            current_sell_value += lot_info['proceeds']
        elif current_sell_value < max_sell_value:
            # Partial sale possible
            remaining_sell_capacity = max_sell_value - current_sell_value
            max_shares_by_value = remaining_sell_capacity / lot_info['lot'].current_price
            
            if max_shares_by_value >= lot_size:
                # Calculate partial sale
                partial_shares = int(max_shares_by_value // lot_size) * lot_size
                partial_proceeds = partial_shares * lot_info['lot'].current_price
                
                # Create modified lot_info for partial sale
                partial_lot_info = lot_info.copy()
                partial_lot_info['shares_to_sell'] = partial_shares
                partial_lot_info['proceeds'] = partial_proceeds
                partial_lot_info['realized_loss'] = partial_shares * lot_info['lot'].gain_loss_per_share
                
                selected_sales.append(partial_lot_info)
                current_sell_value += partial_proceeds
    
    # Create actions for all lots
    for lot_index, lot in enumerate(tax_lots):
        # Check if this lot was selected for sale
        selected_sale = next((s for s in selected_sales if s['lot_index'] == lot_index), None)
        
        if selected_sale:
            actions.append(TLHAction(
                ticker=lot.ticker,
                action="SELL",
                lot_index=lot_index,
                shares_to_sell=selected_sale['shares_to_sell'],
                cost_basis=lot.cost_basis,
                current_price=lot.current_price,
                realized_loss=selected_sale['realized_loss'],
                purchase_date=lot.purchase_date
            ))
        else:
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
    
    # Calculate final tracking error impact
    final_weights = current_weights.copy()
    for sale in selected_sales:
        lot = sale['lot']
        proceeds = sale['proceeds']
        final_weights[lot.ticker] = max(0, (current_weights.get(lot.ticker, 0) * total_portfolio_value - proceeds) / total_portfolio_value)
    
    tracking_error_impact = calculate_tracking_error_impact(current_weights, final_weights, target_weights)
    
    # Calculate summary metrics
    total_realized_losses = sum(action.realized_loss for action in actions if action.action == "SELL")
    total_proceeds = sum(action.proceeds for action in actions if action.action == "SELL")
    total_shares_sold = sum(action.shares_to_sell for action in actions if action.action == "SELL")
    portfolio_impact_percentage = (current_sell_value / total_portfolio_value) * 100
    
    return TLHResult(
        total_portfolio_value=total_portfolio_value,
        lot_size=lot_size,
        max_sell_percentage=max_sell_percentage,
        index_name=index_name.lower(),
        actions=actions,
        total_realized_losses=total_realized_losses,
        total_proceeds=total_proceeds,
        total_shares_sold=total_shares_sold,
        portfolio_impact_percentage=portfolio_impact_percentage,
        tracking_error_impact=tracking_error_impact
    )


def tax_loss_harvest(tax_lots: List[Dict[str, Any]],
                    index_name: Optional[str] = None,
                    lot_size: float = 1.0,
                    max_sell_percentage: float = 100.0) -> Dict[str, Any]:
    """
    Core tax loss harvesting function with automatic strategy selection.
    
    Args:
        tax_lots: List of tax lot dictionaries with keys:
            - ticker: Stock symbol
            - shares: Number of shares
            - cost_basis: Cost per share
            - purchase_date: Purchase date (YYYY-MM-DD string or date object)
        index_name: Optional target index name for tracking error minimization
        lot_size: Minimum trading lot size in shares
        max_sell_percentage: Maximum percentage of portfolio value to sell
        
    Returns:
        Dictionary containing TLH analysis results
    """
    try:
        # Validate and convert tax lots
        validated_lots = validate_tax_lots(tax_lots)
        
        # Select strategy based on index_name parameter
        if index_name:
            result = _tax_loss_harvest_index_optimized(
                validated_lots, index_name, lot_size, max_sell_percentage
            )
        else:
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
    fifo_result = tax_loss_harvest(test_tax_lots, lot_size=10.0, max_sell_percentage=50.0)
    print(format_tlh_summary(fifo_result))
    print()
    
    # Test 2: Index-optimized TLH
    print("2. Testing Index-Optimized Tax Loss Harvesting:")
    index_result = tax_loss_harvest(test_tax_lots, index_name="index1", lot_size=10.0, max_sell_percentage=50.0)
    print(format_tlh_summary(index_result))