# Portfolio Agent

A modular agent framework for portfolio management and financial tools.

## Architecture

The project uses a modular architecture with two main components:

- **`langchainagent/`** - Contains generic agent framework and infrastructure
- **`finance/`** - Contains portfolio-specific implementations and tools

## Quick Start

### 1. Testing Portfolio Logic

Run the following scripts to verify the portfolio math and logic is working correctly:

```bash
# Test tax loss harvesting logic
python -m finance.tests.test_tax_lot_harvest

# Run minimal tests
python -m finance.tests.test_minimal
```

### 2. Demo React Agent

Run the portfolio tools demo with React agent integration:

```bash
python -m finance.demo_finance_react_main
```

*Note: This demo also includes weather and temperature tools for demonstration purposes only.*

### 3. Finance Agent Server

Start the portfolio management server:

#### Option 1: Direct Server Start
```bash
python -m finance.agent.portfolio_server
```

#### Option 2: Test Client Connection
```bash
python -m finance.tests.test_portfolio_client
```

## Project Structure

```
PortfolioAgent/
├── langchainagent/          # Generic agent framework
│   ├── agent_config.py      # Agent configuration protocol
│   └── server.py            # Generic server implementation
├── finance/                 # Portfolio domain implementation
│   ├── agent/              # Portfolio agent configuration
│   ├── tools/              # Financial calculation tools
│   └── tests/              # Test suites
└── README.md               # This file
```

## Features

- **Tax Loss Harvesting** - Optimize portfolio for tax efficiency
- **Portfolio Rebalancing** - Maintain target allocations
- **Index Analysis** - Compare portfolio performance to benchmarks  
- **Stock Price Lookup** - Real-time market data integration
- **Modular Architecture** - Easy to extend with new domains