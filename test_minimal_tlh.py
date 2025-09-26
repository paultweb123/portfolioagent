#!/usr/bin/env python3
"""
Minimal test script for tax loss harvesting function
Tests input tax lots vs harvesting output - ultra-minimal version
"""

import unittest
from portfolio_tlh import (
    tax_loss_harvest,
    format_tlh_summary
)


class TestMinimalTLH(unittest.TestCase):
    """Minimal test class for tax loss harvesting."""
    
    def test_minimal_tlh_fifo(self):
       
        
        # Two tax lots: one profitable, one at significant loss that should be harvested
        input_tax_lots = [
            {
                "ticker": "AAPL",
                "shares": 50.0,  # Smaller position
                "cost_basis": 150.0,  # Profit: current $185 vs cost $150 = $35/share profit
                "purchase_date": "2023-01-15"
            },
            {
                "ticker": "TSLA",
                "shares": 500.0,  # Large loss position
                "cost_basis": 300.0,  # Big Loss: current $220 vs cost $300 = $80/share loss
                "purchase_date": "2023-02-10"
            }
        ]

        
        result = tax_loss_harvest.invoke({
            "tax_lots": input_tax_lots,
            "index_name": None,  # Use FIFO strategy for predictable loss harvesting
            "lot_size": 1.0,
            "max_sell_percentage": 50.0  # Max 25% of portfolio
        })
        
      
        # Print results for inspection
        print(format_tlh_summary(result))
        reference_result = [
            {
                'ticker': 'AAPL',
                'lot_index': 0,
                'action': 'HOLD',
                'shares_to_sell': 0.0
            },
            {
                'ticker': 'TSLA',
                'lot_index': 1,
                'action': 'SELL',
                'shares_to_sell': 250.0
            }
        ]

        extracted_actions = [
                {
                    'ticker': action['ticker'],
                    'lot_index': action['lot_index'],
                    'action': action['action'],
                    'shares_to_sell': action['shares_to_sell']
                }
                for action in result['actions']
            ]
        
        self.assertEqual(reference_result, extracted_actions)

    def test_minimal_tlh_index(self):
       
        
        # Two tax lots: one profitable, one at significant loss that should be harvested
        input_tax_lots = [
            {
                "ticker": "ABC",
                "shares": 100.0,  
                "cost_basis": 80.0,  
                "purchase_date": "2023-01-15"
            },
            {
                "ticker": "XYZ",
                "shares": 100,  
                "cost_basis": 120.0,  
                "purchase_date": "2023-02-10"
            }
        ]

        
        result = tax_loss_harvest.invoke({
            "tax_lots": input_tax_lots,
            "index_name": "Index3",
            "lot_size": 1.0,
            "allocation_tolerance": 5.0,
            "verbose": True
        })
        
      
        # Print results for inspection
        print(format_tlh_summary(result))
      
        reference_result = [
            {
                'ticker': 'ABC',
                'lot_index': 0,
                'action': 'HOLD',
                'shares_to_sell': 0.0
            },
            {
                'ticker': 'XYZ',
                'lot_index': 1,
                'action': 'SELL',
                'shares_to_sell': 70.0
            }
        ]

        extracted_actions = [
                {
                    'ticker': action['ticker'],
                    'lot_index': action['lot_index'],
                    'action': action['action'],
                    'shares_to_sell': action['shares_to_sell']
                }
                for action in result['actions']
            ]
        from pprint import pformat
        print("reference_result\n", pformat(reference_result))
        print("extracted_actions\n", pformat(extracted_actions))
        
        
        self.assertEqual(reference_result, extracted_actions)

if __name__ == "__main__":
    unittest.main()