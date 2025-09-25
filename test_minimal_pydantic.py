#!/usr/bin/env python3
"""
Minimal test script for Pydantic portfolio rebalancing function
Tests input dictionary vs current holdings output using Pydantic models
"""

import unittest
from portfolio_tools_pydantic import (
    pydantic_rebalance_portfolio,
    RebalancingRequest,
    IndexName
)


class TestMinimalPydantic(unittest.TestCase):
    """Minimal test class for Pydantic portfolio rebalancing."""
    
    def test_minimal_comparison_lot1(self):
        """Ultra-minimal: input dict vs current holdings dict using Pydantic models."""
        # Create type-safe request using Pydantic models
        request = RebalancingRequest(
            holdings={"AAPL": 100.0},
            index_name=IndexName.INDEX1,
            lot_size=1.0
        )
        
        # With lot-based trading (lot_size=1.0), expect final lot-rounded shares
        expected_output_holdings = {
            "AAPL": 30.0,   # Exact target: 30.0
            "AMZN": 19.0,   # 19.137931... → 19.0 lots
            "GOOGL": 1.0,   # 1.6517857... → 1.0 lots
            "MSFT": 9.0,    # 9.866666... → 9.0 lots
            "TSLA": 8.0     # 8.409090... → 8.0 lots
        }
        
        result = pydantic_rebalance_portfolio(request)
        output_current = {h.ticker: h.shares for h in result.target_holdings}

        self.assertEqual(expected_output_holdings, output_current)

        # Convert back to dict format for printing with original formatter
        result_dict = {
            "target_index": result.target_index,
            "current_portfolio_value": result.current_portfolio_value,
            "lot_size": result.lot_size,
            "current_holdings": [h.model_dump() for h in result.current_holdings],
            "rebalancing_actions": [a.model_dump() for a in result.rebalancing_actions],
            "target_holdings": [h.model_dump() for h in result.target_holdings],
            "total_trading_value": result.total_trading_value,
            "number_of_actions": result.number_of_actions
        }
        
        from portfolio_tools import format_rebalancing_summary
        print(format_rebalancing_summary(result_dict))
        


    def test_minimal_comparison_lot10(self):
        """Ultra-minimal: input dict vs current holdings dict using Pydantic models."""
        # Create type-safe request using Pydantic models
        request = RebalancingRequest(
            holdings={"AAPL": 1000.0},
            index_name=IndexName.INDEX1,
            lot_size=10.0
        )
        
        # With lot-based trading (lot_size=10.0), expect final lot-rounded shares
        expected_output_holdings = {'AMZN': 190.0, 'GOOGL': 10.0, 'MSFT': 90.0, 'TSLA': 80.0, 'AAPL': 300.0}
        
        result = pydantic_rebalance_portfolio(request)
        output_current = {h.ticker: h.shares for h in result.target_holdings}

        self.assertEqual(expected_output_holdings, output_current)

        # Convert back to dict format for printing with original formatter
        result_dict = {
            "target_index": result.target_index,
            "current_portfolio_value": result.current_portfolio_value,
            "lot_size": result.lot_size,
            "current_holdings": [h.model_dump() for h in result.current_holdings],
            "rebalancing_actions": [a.model_dump() for a in result.rebalancing_actions],
            "target_holdings": [h.model_dump() for h in result.target_holdings],
            "total_trading_value": result.total_trading_value,
            "number_of_actions": result.number_of_actions
        }
        
        from portfolio_tools import format_rebalancing_summary
        print(format_rebalancing_summary(result_dict))



    def test_minimal_comparison_notlotset(self):
        """Ultra-minimal: input dict vs current holdings dict using Pydantic models."""
        # Create type-safe request using Pydantic models
        request = RebalancingRequest(
            holdings={"AAPL": 100.0},
            index_name=IndexName.INDEX1,
            lot_size=1.0  # Explicitly provide default lot_size
        )
        
        # With lot-based trading (default lot_size=1.0), expect final lot-rounded shares
        expected_output_holdings = {
            "AAPL": 30.0,   # Exact target: 30.0
            "AMZN": 19.0,   # 19.137931... → 19.0 lots
            "GOOGL": 1.0,   # 1.6517857... → 1.0 lots
            "MSFT": 9.0,    # 9.866666... → 9.0 lots
            "TSLA": 8.0     # 8.409090... → 8.0 lots
        }
        
        result = pydantic_rebalance_portfolio(request)
        output_current = {h.ticker: h.shares for h in result.target_holdings}

        self.assertEqual(expected_output_holdings, output_current)

        # Convert back to dict format for printing with original formatter
        result_dict = {
            "target_index": result.target_index,
            "current_portfolio_value": result.current_portfolio_value,
            "lot_size": result.lot_size,
            "current_holdings": [h.model_dump() for h in result.current_holdings],
            "rebalancing_actions": [a.model_dump() for a in result.rebalancing_actions],
            "target_holdings": [h.model_dump() for h in result.target_holdings],
            "total_trading_value": result.total_trading_value,
            "number_of_actions": result.number_of_actions
        }
        
        from portfolio_tools import format_rebalancing_summary
        print(format_rebalancing_summary(result_dict))
        


if __name__ == "__main__":
    unittest.main()