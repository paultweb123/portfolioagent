#!/usr/bin/env python3
"""
Test script for tax loss harvesting functions
Tests FIFO and index-optimized TLH strategies with various scenarios
"""

import unittest
from datetime import date
from portfolio_tlh import (
    tax_loss_harvest,
    format_tlh_summary,
    validate_tax_lots,
    TaxLot
)


class TestTaxLossHarvesting(unittest.TestCase):
    """Test class for tax loss harvesting functionality."""
    
    def setUp(self):
        """Set up common test data."""
        # Test data with mix of gains and losses
        self.test_tax_lots = [
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
            },
            {
                "ticker": "TSLA",
                "shares": 30.0,
                "cost_basis": 250.0,  # Loss: current $220 vs cost $250
                "purchase_date": "2023-04-10"
            }
        ]
        
        # All-loss portfolio for maximum harvesting tests
        self.all_loss_lots = [
            {
                "ticker": "AAPL",
                "shares": 100.0,
                "cost_basis": 250.0,  # Loss: current $185 vs cost $250
                "purchase_date": "2023-01-15"
            },
            {
                "ticker": "GOOGL",
                "shares": 5.0,
                "cost_basis": 3200.0,  # Loss: current $2800 vs cost $3200
                "purchase_date": "2023-02-10"
            },
            {
                "ticker": "MSFT",
                "shares": 20.0,
                "cost_basis": 450.0,  # Loss: current $375 vs cost $450
                "purchase_date": "2023-03-05"
            }
        ]

    def test_fifo_basic_functionality(self):
        """Test basic FIFO tax loss harvesting."""
        result = tax_loss_harvest(
            self.test_tax_lots,
            lot_size=10.0,
            max_sell_percentage=100.0
        )
        
        # Should use FIFO strategy
        self.assertEqual(result["strategy"], "fifo")
        self.assertIsNone(result["index_name"])
        
        # Should have positive portfolio value
        self.assertGreater(result["total_portfolio_value"], 0)
        
        # Should have some realized losses (there are loss positions)
        self.assertLess(result["total_realized_losses"], 0)  # Losses are negative
        
        # Should have some sales
        self.assertGreater(result["number_of_sales"], 0)
        
        print(f"\nFIFO Basic Test Results:")
        print(format_tlh_summary(result))

    def test_fifo_lot_size_constraints(self):
        """Test FIFO with different lot size constraints."""
        # Test with large lot size that should limit trades
        result_large_lot = tax_loss_harvest(
            self.test_tax_lots,
            lot_size=100.0,  # Large lot size
            max_sell_percentage=100.0
        )
        
        # Test with small lot size that should allow more trades
        result_small_lot = tax_loss_harvest(
            self.test_tax_lots,
            lot_size=1.0,   # Small lot size
            max_sell_percentage=100.0
        )
        
        # Small lot size should generally allow more or equal sales
        self.assertGreaterEqual(
            result_small_lot["number_of_sales"],
            result_large_lot["number_of_sales"]
        )
        
        print(f"\nFIFO Lot Size Test - Large lot ({result_large_lot['number_of_sales']} sales):")
        print(f"Total losses: ${result_large_lot['total_realized_losses']:,.2f}")
        print(f"\nFIFO Lot Size Test - Small lot ({result_small_lot['number_of_sales']} sales):")
        print(f"Total losses: ${result_small_lot['total_realized_losses']:,.2f}")

    def test_fifo_sell_percentage_limits(self):
        """Test FIFO with sell percentage limits."""
        # Test with 25% limit
        result_25 = tax_loss_harvest(
            self.test_tax_lots,
            lot_size=1.0,
            max_sell_percentage=25.0
        )
        
        # Test with 75% limit
        result_75 = tax_loss_harvest(
            self.test_tax_lots,
            lot_size=1.0,
            max_sell_percentage=75.0
        )
        
        # Should respect percentage limits
        self.assertLessEqual(result_25["portfolio_impact_percentage"], 25.1)  # Allow small rounding
        self.assertLessEqual(result_75["portfolio_impact_percentage"], 75.1)
        
        # Higher percentage should generally allow more harvesting
        self.assertGreaterEqual(
            abs(result_75["total_realized_losses"]),
            abs(result_25["total_realized_losses"])
        )
        
        print(f"\nFIFO Sell Limits - 25%: ${result_25['total_realized_losses']:,.2f} losses")
        print(f"FIFO Sell Limits - 75%: ${result_75['total_realized_losses']:,.2f} losses")

    def test_index_optimized_basic_functionality(self):
        """Test basic index-optimized tax loss harvesting."""
        result = tax_loss_harvest(
            self.test_tax_lots,
            index_name="index1",
            lot_size=10.0,
            max_sell_percentage=100.0
        )
        
        # Should use index-optimized strategy
        self.assertEqual(result["strategy"], "index_optimized")
        self.assertEqual(result["index_name"], "index1")
        
        # Should have tracking error impact calculated
        self.assertIsNotNone(result["tracking_error_impact"])
        
        # Should have positive portfolio value
        self.assertGreater(result["total_portfolio_value"], 0)
        
        print(f"\nIndex-Optimized Basic Test Results:")
        print(format_tlh_summary(result))

    def test_index_optimized_vs_fifo_comparison(self):
        """Compare index-optimized vs FIFO strategies."""
        fifo_result = tax_loss_harvest(
            self.test_tax_lots,
            lot_size=10.0,
            max_sell_percentage=50.0
        )
        
        index_result = tax_loss_harvest(
            self.test_tax_lots,
            index_name="index1",
            lot_size=10.0,
            max_sell_percentage=50.0
        )
        
        # Both should have same portfolio value
        self.assertAlmostEqual(
            fifo_result["total_portfolio_value"],
            index_result["total_portfolio_value"],
            places=2
        )
        
        # Index-optimized should have tracking error impact
        self.assertIsNone(fifo_result["tracking_error_impact"])
        self.assertIsNotNone(index_result["tracking_error_impact"])
        
        print(f"\nStrategy Comparison:")
        print(f"FIFO: {fifo_result['number_of_sales']} sales, ${fifo_result['total_realized_losses']:,.2f} losses")
        print(f"Index: {index_result['number_of_sales']} sales, ${index_result['total_realized_losses']:,.2f} losses")
        print(f"Tracking Error Impact: {index_result['tracking_error_impact']:.6f}")

    def test_all_loss_portfolio_maximum_harvesting(self):
        """Test maximum harvesting with all-loss portfolio."""
        result = tax_loss_harvest(
            self.all_loss_lots,
            lot_size=1.0,
            max_sell_percentage=100.0
        )
        
        # Should harvest losses from all positions
        self.assertGreater(result["number_of_sales"], 0)
        self.assertLess(result["total_realized_losses"], 0)  # Should be negative (losses)
        
        # All actions should be either SELL or HOLD (no gains to skip)
        for action in result["actions"]:
            self.assertIn(action["action"], ["SELL", "HOLD"])
        
        print(f"\nAll-Loss Portfolio Test:")
        print(format_tlh_summary(result))

    def test_edge_case_empty_lots(self):
        """Test error handling for empty tax lots."""
        result = tax_loss_harvest([])
        self.assertIn("error", result)
        self.assertEqual(result["strategy"], "error")

    def test_edge_case_invalid_lot_size(self):
        """Test error handling for invalid lot size."""
        result1 = tax_loss_harvest(self.test_tax_lots, lot_size=0.0)
        self.assertIn("error", result1)
        self.assertEqual(result1["strategy"], "error")
        
        result2 = tax_loss_harvest(self.test_tax_lots, lot_size=-1.0)
        self.assertIn("error", result2)
        self.assertEqual(result2["strategy"], "error")

    def test_edge_case_invalid_sell_percentage(self):
        """Test error handling for invalid sell percentage."""
        result1 = tax_loss_harvest(self.test_tax_lots, max_sell_percentage=0.0)
        self.assertIn("error", result1)
        self.assertEqual(result1["strategy"], "error")
        
        result2 = tax_loss_harvest(self.test_tax_lots, max_sell_percentage=101.0)
        self.assertIn("error", result2)
        self.assertEqual(result2["strategy"], "error")

    def test_edge_case_invalid_index(self):
        """Test error handling for invalid index name."""
        result = tax_loss_harvest(self.test_tax_lots, index_name="invalid_index")
        self.assertIn("error", result)
        self.assertEqual(result["strategy"], "error")

    def test_tax_lot_validation(self):
        """Test tax lot validation functionality."""
        # Valid tax lots
        valid_lots = [
            {
                "ticker": "AAPL",
                "shares": 100.0,
                "cost_basis": 200.0,
                "purchase_date": "2023-01-15"
            }
        ]
        validated = validate_tax_lots(valid_lots)
        self.assertEqual(len(validated), 1)
        self.assertIsInstance(validated[0], TaxLot)
        
        # Invalid tax lots - missing ticker
        invalid_lots = [
            {
                "shares": 100.0,
                "cost_basis": 200.0,
                "purchase_date": "2023-01-15"
            }
        ]
        with self.assertRaises(ValueError):
            validate_tax_lots(invalid_lots)

    def test_different_index_targets(self):
        """Test with different index targets."""
        index1_result = tax_loss_harvest(
            self.test_tax_lots,
            index_name="index1",
            lot_size=10.0,
            max_sell_percentage=50.0
        )
        
        index2_result = tax_loss_harvest(
            self.test_tax_lots,
            index_name="index2", 
            lot_size=10.0,
            max_sell_percentage=50.0
        )
        
        # Both should be valid results
        self.assertEqual(index1_result["strategy"], "index_optimized")
        self.assertEqual(index2_result["strategy"], "index_optimized")
        self.assertEqual(index1_result["index_name"], "index1")
        self.assertEqual(index2_result["index_name"], "index2")
        
        # May have different tracking error impacts due to different target weights
        self.assertIsNotNone(index1_result["tracking_error_impact"])
        self.assertIsNotNone(index2_result["tracking_error_impact"])
        
        print(f"\nIndex Target Comparison:")
        print(f"Index1: ${index1_result['total_realized_losses']:,.2f} losses, tracking error: {index1_result['tracking_error_impact']:.6f}")
        print(f"Index2: ${index2_result['total_realized_losses']:,.2f} losses, tracking error: {index2_result['tracking_error_impact']:.6f}")

    def test_fifo_chronological_order(self):
        """Test that FIFO respects chronological order."""
        # Create lots with clear chronological order and different loss amounts
        chronological_lots = [
            {
                "ticker": "AAPL",
                "shares": 50.0,
                "cost_basis": 300.0,  # Big loss: current $185 vs cost $300
                "purchase_date": "2023-01-01"  # Oldest
            },
            {
                "ticker": "AAPL",
                "shares": 50.0,
                "cost_basis": 250.0,  # Medium loss: current $185 vs cost $250
                "purchase_date": "2023-06-01"  # Middle
            },
            {
                "ticker": "AAPL",
                "shares": 50.0,
                "cost_basis": 200.0,  # Small loss: current $185 vs cost $200
                "purchase_date": "2023-12-01"  # Newest
            }
        ]
        
        result = tax_loss_harvest(
            chronological_lots,
            lot_size=10.0,
            max_sell_percentage=50.0  # Limited to force selection
        )
        
        # Check that sales respect FIFO order
        sell_actions = [action for action in result["actions"] if action["action"] == "SELL"]
        if len(sell_actions) > 1:
            # Should sell older lots first
            dates = [action["purchase_date"] for action in sell_actions]
            self.assertEqual(dates, sorted(dates))
        
        print(f"\nFIFO Chronological Test:")
        for action in result["actions"]:
            if action["action"] == "SELL":
                print(f"SELL {action['shares_to_sell']} shares from {action['purchase_date']} (${action['realized_loss']:.2f} loss)")

    def test_return_data_structure(self):
        """Test that returned data structure matches expected format."""
        result = tax_loss_harvest(self.test_tax_lots, lot_size=10.0)
        
        # Check required top-level keys
        required_keys = [
            "strategy", "total_portfolio_value", "lot_size", "max_sell_percentage",
            "index_name", "total_realized_losses", "total_proceeds", "total_shares_sold",
            "number_of_sales", "portfolio_impact_percentage", "tracking_error_impact", "actions"
        ]
        
        for key in required_keys:
            self.assertIn(key, result, f"Missing required key: {key}")
        
        # Check action structure
        if result["actions"]:
            action = result["actions"][0]
            action_keys = [
                "ticker", "action", "lot_index", "shares_to_sell", "cost_basis",
                "current_price", "realized_loss", "purchase_date", "proceeds", "total_cost_basis"
            ]
            for key in action_keys:
                self.assertIn(key, action, f"Missing action key: {key}")


if __name__ == "__main__":
    unittest.main(verbosity=2)