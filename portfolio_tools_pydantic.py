#!/usr/bin/env python3
"""
Pydantic Portfolio Rebalancing Tools for React Agents
This module provides type-safe Pydantic models wrapping the portfolio_tools functionality.
"""

from typing import Dict, List
from enum import Enum
from pydantic import BaseModel, Field, validator
from portfolio_tools import rebalance_portfolio as original_rebalance_portfolio


class IndexName(str, Enum):
    """Valid index names for portfolio rebalancing."""
    INDEX1 = "index1"
    INDEX2 = "index2"


class ActionType(str, Enum):
    """Valid action types for rebalancing trades."""
    BUY = "BUY"
    SELL = "SELL" 
    HOLD = "HOLD"


class PortfolioHolding(BaseModel):
    """
    Represents a single portfolio holding with market data.
    
    Attributes:
        ticker: Stock symbol (e.g., "AAPL")
        shares: Number of shares held
        price: Current stock price per share
        market_value: Total position value (shares * price)
        weight_percent: Position weight as percentage (0-100)
    """
    ticker: str = Field(..., description="Stock ticker symbol")
    shares: float = Field(..., ge=0, description="Number of shares")
    price: float = Field(..., gt=0, description="Stock price per share")
    market_value: float = Field(..., ge=0, description="Total market value")
    weight_percent: float = Field(..., ge=0, le=100, description="Weight percentage")

    @validator('ticker')
    def ticker_must_be_uppercase_string(cls, v):
        if not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError('Ticker must be a non-empty string')
        return v.upper().strip()


class RebalancingAction(BaseModel):
    """
    Represents a rebalancing trade recommendation.
    
    Attributes:
        ticker: Stock symbol
        action: Trade action (BUY, SELL, or HOLD)
        current_weight_percent: Current allocation percentage
        target_weight_percent: Target allocation percentage
        shares: Number of shares to trade (0 for HOLD)
        amount: Dollar amount of trade (0 for HOLD)
    """
    ticker: str = Field(..., description="Stock ticker symbol")
    action: ActionType = Field(..., description="Trade action")
    current_weight_percent: float = Field(..., ge=0, le=100, description="Current weight %")
    target_weight_percent: float = Field(..., ge=0, le=100, description="Target weight %")
    shares: float = Field(..., ge=0, description="Shares to trade")
    amount: float = Field(..., ge=0, description="Trade amount in dollars")

    @validator('ticker')
    def ticker_must_be_uppercase_string(cls, v):
        if not isinstance(v, str) or len(v.strip()) == 0:
            raise ValueError('Ticker must be a non-empty string')
        return v.upper().strip()


class RebalancingRequest(BaseModel):
    """
    Input parameters for portfolio rebalancing.
    
    Attributes:
        holdings: Dictionary mapping ticker symbols to share quantities
        index_name: Target index name ("index1" or "index2")
        lot_size: Minimum trading lot size in shares (default: 1.0)
    """
    holdings: Dict[str, float] = Field(..., description="Portfolio holdings by ticker")
    index_name: IndexName = Field(..., description="Target index")
    lot_size: float = Field(default=1.0, gt=0, description="Trading lot size")

    @validator('holdings')
    def holdings_must_be_valid(cls, v):
        if not v:
            raise ValueError('Portfolio must have at least one holding')
        
        validated_holdings = {}
        for ticker, quantity in v.items():
            if not isinstance(ticker, str) or len(ticker.strip()) == 0:
                raise ValueError(f'Invalid ticker: {ticker}')
            if not isinstance(quantity, (int, float)) or quantity < 0:
                raise ValueError(f'Invalid quantity for {ticker}: {quantity}')
            validated_holdings[ticker.upper().strip()] = float(quantity)
        
        return validated_holdings


class RebalancingResult(BaseModel):
    """
    Complete portfolio rebalancing analysis results.
    
    Attributes:
        target_index: Target index name used for rebalancing
        current_portfolio_value: Total market value of current portfolio
        lot_size: Lot size used for trading calculations
        current_holdings: List of current portfolio positions
        rebalancing_actions: List of recommended trading actions
        target_holdings: List of final portfolio positions after rebalancing
        total_trading_value: Total dollar value of all trades
        number_of_actions: Count of non-HOLD actions required
    """
    target_index: str = Field(..., description="Target index name")
    current_portfolio_value: float = Field(..., ge=0, description="Current portfolio value")
    lot_size: float = Field(..., gt=0, description="Trading lot size")
    current_holdings: List[PortfolioHolding] = Field(..., description="Current positions")
    rebalancing_actions: List[RebalancingAction] = Field(..., description="Trade recommendations")
    target_holdings: List[PortfolioHolding] = Field(..., description="Final positions")
    total_trading_value: float = Field(..., ge=0, description="Total trade value")
    number_of_actions: int = Field(..., ge=0, description="Number of actions")


def pydantic_rebalance_portfolio(request: RebalancingRequest) -> RebalancingResult:
    """
    Type-safe wrapper for portfolio rebalancing using Pydantic models.
    
    This function provides the same functionality as the original rebalance_portfolio
    but with full type safety, validation, and structured data models suitable for
    React agents and other automated systems.
    
    Args:
        request: RebalancingRequest containing holdings, index_name, and lot_size
        
    Returns:
        RebalancingResult: Structured analysis with type-safe data models
        
    Example:
        >>> request = RebalancingRequest(
        ...     holdings={"AAPL": 100.0, "GOOGL": 25.0},
        ...     index_name=IndexName.INDEX1,
        ...     lot_size=10.0
        ... )
        >>> result = pydantic_rebalance_portfolio(request)
        >>> print(f"Portfolio value: ${result.current_portfolio_value:,.2f}")
        >>> for action in result.rebalancing_actions:
        ...     if action.action != ActionType.HOLD:
        ...         print(f"{action.action.value} {action.shares} shares of {action.ticker}")
    """
    # Convert Pydantic request to dict format for original function
    result_dict = original_rebalance_portfolio.invoke({
        "holdings": request.holdings,
        "index_name": request.index_name.value,
        "lot_size": request.lot_size
    })
    
    # Convert dict result to Pydantic models
    current_holdings = [
        PortfolioHolding(
            ticker=h["ticker"],
            shares=h["shares"],
            price=h["price"],
            market_value=h["market_value"],
            weight_percent=h["weight_percent"]
        ) for h in result_dict["current_holdings"]
    ]
    
    rebalancing_actions = [
        RebalancingAction(
            ticker=a["ticker"],
            action=ActionType(a["action"]),
            current_weight_percent=a["current_weight_percent"],
            target_weight_percent=a["target_weight_percent"],
            shares=a["shares"],
            amount=a["amount"]
        ) for a in result_dict["rebalancing_actions"]
    ]
    
    target_holdings = [
        PortfolioHolding(
            ticker=h["ticker"],
            shares=h["shares"],
            price=h["price"],
            market_value=h["market_value"],
            weight_percent=h["weight_percent"]
        ) for h in result_dict["target_holdings"]
    ]
    
    return RebalancingResult(
        target_index=result_dict["target_index"],
        current_portfolio_value=result_dict["current_portfolio_value"],
        lot_size=result_dict["lot_size"],
        current_holdings=current_holdings,
        rebalancing_actions=rebalancing_actions,
        target_holdings=target_holdings,
        total_trading_value=result_dict["total_trading_value"],
        number_of_actions=result_dict["number_of_actions"]
    )


# Helper function to get Pydantic models for React agents
def get_pydantic_models():
    """Return all Pydantic models for React agent integration."""
    return {
        "RebalancingRequest": RebalancingRequest,
        "RebalancingResult": RebalancingResult,
        "PortfolioHolding": PortfolioHolding,
        "RebalancingAction": RebalancingAction,
        "IndexName": IndexName,
        "ActionType": ActionType
    }


if __name__ == "__main__":
    # Quick test of Pydantic models
    print("🧪 Testing Pydantic Portfolio Tools\n")
    
    # Test with type-safe request
    request = RebalancingRequest(
        holdings={"AAPL": 100.0, "GOOGL": 20.0, "MSFT": 50.0},
        index_name=IndexName.INDEX1,
        lot_size=10.0
    )
    
    print(f"Request: {request.dict()}")
    
    result = pydantic_rebalance_portfolio(request)
    
    print(f"\n📊 Pydantic Result:")
    print(f"Target Index: {result.target_index}")
    print(f"Portfolio Value: ${result.current_portfolio_value:,.2f}")
    print(f"Trading Actions: {result.number_of_actions}")
    print(f"Trading Value: ${result.total_trading_value:,.2f}")
    
    print("\n🎯 Target Holdings:")
    for holding in result.target_holdings:
        print(f"{holding.ticker}: {holding.shares} shares = ${holding.market_value:,.2f}")