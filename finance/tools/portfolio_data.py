
from langchain_core.tools import tool


# Hardcoded Index Definitions
INDEX_WEIGHTS = {
    "index1": {
        "AAPL": 0.30,    # 30%
        "GOOGL": 0.25,   # 25%
        "MSFT": 0.20,    # 20%
        "AMZN": 0.15,    # 15%
        "TSLA": 0.10     # 10%
    },
    "index2": {
        "AAPL": 0.15,    # 15%
        "GOOGL": 0.12,   # 12%
        "MSFT": 0.10,    # 10%
        "AMZN": 0.08,    # 8%
        "TSLA": 0.08,    # 8%
        "NVDA": 0.12,    # 12%
        "META": 0.10,    # 10%
        "NFLX": 0.08,    # 8%
        "ADBE": 0.09,    # 9%
        "CRM": 0.08      # 8%
    },
    "index3": {
        "ABC": 0.2, 
        "XYZ": 0.2,  
        "PQR": 0.2,  
        "LMN": 0.2,    
        "EFG": 0.2     
    },

}

# Hardcoded Stock Prices
STOCK_PRICES = {
    "AAPL": 185.00,
    "GOOGL": 2800.00,
    "MSFT": 375.00,
    "AMZN": 145.00,
    "TSLA": 220.00,
    "NVDA": 450.00,
    "META": 315.00,
    "NFLX": 425.00,
    "ADBE": 520.00,
    "CRM": 185.00,
    "ABC": 100.00,
    "XYZ": 100.00,
    "PQR": 100.00,
    "LMN": 100.00,
    "EFG": 100.00
}


def validate_index_name(index_name: str) -> str:
    """Validate index name."""
    index_name_clean = index_name.lower().strip()
    if index_name_clean not in ["index1", "index2"]:
        raise ValueError(f"Invalid index name '{index_name}'. Must be 'index1' or 'index2'")
    return index_name_clean

def validate_ticker(ticker: str) -> str:
    """Validate ticker symbol."""
    if not isinstance(ticker, str) or len(ticker.strip()) == 0:
        raise ValueError("Ticker must be a non-empty string")
    return ticker.upper().strip()


@tool
def get_index_weights(index_name: str) -> str:
    """
    Get the target weights for a specific index.
    
    Args:
        index_name: Name of the index ('index1' or 'index2')
        
    Returns:
        Formatted string showing ticker weights for the index
    """
    # Validate using dictionary validation function
    try:
        validated_index_name = validate_index_name(index_name)
    except ValueError as e:
        return f"Error: {str(e)}"
    
    weights = INDEX_WEIGHTS[validated_index_name]
    
    result = f"📊 {validated_index_name.upper()} Target Weights:\n"
    result += "=" * 30 + "\n"
    
    total_weight = 0
    for ticker, weight in weights.items():
        result += f"{ticker:<6}: {weight:>6.1%}\n"
        total_weight += weight
    
    result += "-" * 30 + "\n"
    result += f"{'Total':<6}: {total_weight:>6.1%}\n"
    result += f"\nIndex contains {len(weights)} constituents"
    
    return result

@tool
def get_prices(ticker: str) -> str:
    """
    Get the current price for a specific ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        
    Returns:
        Current price of the ticker as formatted string
    """
    # Validate using dictionary validation function
    try:
        validated_ticker = validate_ticker(ticker)
    except ValueError as e:
        return f"Error: {str(e)}"
    
    if validated_ticker not in STOCK_PRICES:
        available_tickers = ", ".join(sorted(STOCK_PRICES.keys()))
        return f"Error: Unknown ticker '{validated_ticker}'. Available tickers: {available_tickers}"
    
    price = STOCK_PRICES[validated_ticker]
    return f"💰 {validated_ticker} current price: ${price:,.2f}"
