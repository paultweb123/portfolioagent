#!/usr/bin/env python3
"""
Portfolio Management Agent Server

Standalone server for financial portfolio management capabilities.
Uses the generic agent server framework with portfolio-specific configuration.

Usage:
    python finance/agent/portfolio_server.py [--host HOST] [--port PORT] [--verbose]

Examples:
    python finance/agent/portfolio_server.py --port 10000
    python finance/agent/portfolio_server.py --host 0.0.0.0 --port 8080 --verbose
"""

import click
import sys
import os
import logging

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from langchainagent.server import create_agent_server
from finance.agent.portfolio_config import PortfolioAgentConfig


@click.command()
@click.option('--host', default='localhost', help='Server host address')
@click.option('--port', default=10000, type=int, help='Server port number')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def main(host: str, port: int, verbose: bool):
    """Launch Portfolio Management Agent Server"""
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        print("🔧 Verbose logging enabled")
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        print("🏦 Portfolio Management Agent Server")
        print("=" * 50)
        
        # Create portfolio configuration
        config = PortfolioAgentConfig()
        print(f"📊 Domain: {config.domain_name}")
        
        # Validate environment before starting
        print("🔍 Validating environment...")
        config.validate_environment()
        print("✅ Environment validation passed")
        
        # Show configuration details
        tools = config.get_tools()
        agent_card = config.get_agent_card(host, port)
        
        print(f"🏷️  Agent: {agent_card.name} v{agent_card.version}")
        print(f"📋 Skills: {len(agent_card.skills)} available")
        for skill in agent_card.skills:
            print(f"   - {skill.name}")
        print(f"🔧 Tools: {len(tools)} loaded")
        print(f"🌐 Server: http://{host}:{port}")
        
        print("\n" + "=" * 50)
        print("🚀 Starting server...")
        
        # Start the server using generic framework
        create_agent_server(config, host, port)
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Check that required environment variables are set in .env file")
        sys.exit(1)
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Check that required dependencies are installed")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()