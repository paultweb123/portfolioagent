
from finance.tools.tax_lot_harvest import tax_loss_harvest
from finance.tools.rebalance import get_index_weights, get_prices, rebalance_portfolio

# Helper function to get all available tools
def get_portfolio_tools():
    """Return list of all portfolio tools for integration with ReAct agent."""
    return [get_index_weights, get_prices, rebalance_portfolio, tax_loss_harvest]
