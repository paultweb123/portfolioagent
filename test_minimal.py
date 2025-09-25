#!/usr/bin/env python3
"""
Minimal test script for portfolio rebalancing function
Tests input dictionary vs current holdings output
"""

import unittest
from portfolio_tools import (
    rebalance_portfolio,
    format_rebalancing_summary
)


class TestMinimal(unittest.TestCase):
    """Minimal test class for portfolio rebalancing."""
    
    def test_minimal_comparison_lot1(self):
        """Ultra-minimal: input dict vs current holdings dict."""
        input_holdings = {"AAPL": 100.0}
        # With lot-based trading (lot_size=1.0), expect final lot-rounded shares
        expected_output_holdings = {
            "AAPL": 30.0,   # Exact target: 30.0
            "AMZN": 19.0,   # 19.137931... → 19.0 lots
            "GOOGL": 1.0,   # 1.6517857... → 1.0 lots
            "MSFT": 9.0,    # 9.866666... → 9.0 lots
            "TSLA": 8.0     # 8.409090... → 8.0 lots
        }
        result = rebalance_portfolio.invoke({
            "holdings": input_holdings,
            "index_name": "index1",
            "lot_size": 1.0
        })
        output_current = {h["ticker"]: h["shares"] for h in result["target_holdings"]}

        self.assertEqual(expected_output_holdings, output_current)

        print(format_rebalancing_summary(result))
        


    def test_minimal_comparison_lot10(self):
        """Ultra-minimal: input dict vs current holdings dict."""
        input_holdings = {"AAPL": 1000.0}
        # With lot-based trading (lot_size=10.0), expect final lot-rounded shares
        expected_output_holdings = {'AMZN': 190.0, 'GOOGL': 10.0, 'MSFT': 90.0, 'TSLA': 80.0, 'AAPL': 300.0}
        result = rebalance_portfolio.invoke({
            "holdings": input_holdings,
            "index_name": "index1",
            "lot_size": 10.0
        })
        output_current = {h["ticker"]: h["shares"] for h in result["target_holdings"]}

        self.assertEqual(expected_output_holdings, output_current)

        print(format_rebalancing_summary(result))



    def test_minimal_comparison_notlotset(self):
        """Ultra-minimal: input dict vs current holdings dict."""
        input_holdings = {"AAPL": 100.0}
        # With lot-based trading (default lot_size=1.0), expect final lot-rounded shares
        expected_output_holdings = {
            "AAPL": 30.0,   # Exact target: 30.0
            "AMZN": 19.0,   # 19.137931... → 19.0 lots
            "GOOGL": 1.0,   # 1.6517857... → 1.0 lots
            "MSFT": 9.0,    # 9.866666... → 9.0 lots
            "TSLA": 8.0     # 8.409090... → 8.0 lots
        }
        result = rebalance_portfolio.invoke({
            "holdings": input_holdings,
            "index_name": "index1",
            "lot_size": 1.0  # Explicitly provide default lot_size
        })
        output_current = {h["ticker"]: h["shares"] for h in result["target_holdings"]}

        self.assertEqual(expected_output_holdings, output_current)

        print(format_rebalancing_summary(result))
        



if __name__ == "__main__":
    unittest.main()