#!/usr/bin/env python3
"""
Portfolio Management Agent Server

Minimal server entry point that uses the generic agent server framework.
All CLI logic, logging, validation, and startup handling is provided by the framework.

Usage:
    python finance/agent/portfolio_server.py [--host HOST] [--port PORT] [--verbose]

Examples:
    python finance/agent/portfolio_server.py --port 10000
    python finance/agent/portfolio_server.py --host 0.0.0.0 --port 8080 --verbose
"""

import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchainagent.server import run_agent_server
from finance.agent.portfolio_config import PortfolioAgentConfig


if __name__ == '__main__':
    run_agent_server(
        config_class=PortfolioAgentConfig,
        agent_description="Portfolio Management Agent", 
        emoji="🏦"
    )