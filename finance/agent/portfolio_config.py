"""
Portfolio Agent Configuration

This module implements the AgentConfiguration protocol for portfolio management,
providing all portfolio-specific skills, tools, and agent card information.
"""

import os
from typing import List
from langchain_core.tools import BaseTool
from a2a.types import AgentCard, AgentSkill, AgentCapabilities
from langchainagent.agent_config import AgentConfiguration
from langchainagent.agent import LangchainReactAgent


class PortfolioAgentConfig:
    """Configuration for Portfolio Management Agent"""
    
    @property
    def domain_name(self) -> str:
        return "finance"
    
    def validate_environment(self) -> None:
        """Validate domain-specific environment variables and dependencies"""
        # Check for required API keys
        if not os.getenv('ANTHROPIC_API_KEY'):
            if os.getenv('model_source', 'google') == 'google':
                if not os.getenv('GOOGLE_API_KEY'):
                    raise ValueError("GOOGLE_API_KEY environment variable required")
            else:
                if not os.getenv('TOOL_LLM_URL'):
                    raise ValueError("TOOL_LLM_URL environment variable required")
                if not os.getenv('TOOL_LLM_NAME'):
                    raise ValueError("TOOL_LLM_NAME environment variable required")
        
        # Validate portfolio tools availability
        try:
            import finance.tools.portfolio_tools
        except ImportError:
            raise ImportError("portfolio_tools module is required for portfolio agent")
    
    def get_tools(self) -> List[BaseTool]:
        """Return the LangChain tools for portfolio management"""
        import finance.tools.portfolio_tools as portfolio_tools
        return portfolio_tools.get_portfolio_tools()
    
    def get_agent_card(self, host: str, port: int) -> AgentCard:
        """Return the agent card for portfolio management"""
        capabilities = AgentCapabilities(streaming=True, push_notifications=True)
        
        # Convert 0.0.0.0 to localhost for client connections
        client_host = 'localhost' if host == '0.0.0.0' else host

        print(f"Creating agent card with host: {client_host}, port: {port}")
        
        return AgentCard(
            name='Portfolio Management Agent',
            description='Comprehensive portfolio management with advanced analytics and tax-optimized strategies',
            url=f'http://{client_host}:{port}/',
            version='2.0.0',
            default_input_modes=LangchainReactAgent.SUPPORTED_CONTENT_TYPES,
            default_output_modes=LangchainReactAgent.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=self._create_portfolio_skills()
        )
    
    def _create_portfolio_skills(self) -> List[AgentSkill]:
        """Create portfolio-specific skills"""
        return [
            AgentSkill(
                id='portfolio_rebalancing',
                name='Portfolio Rebalancing',
                description='Rebalances portfolios to match target index allocations with lot-based trading',
                tags=['portfolio management', 'rebalancing', 'index tracking', 'asset allocation'],
                examples=[
                    'Rebalance my portfolio to match index1 with 10 share lots',
                    'How should I rebalance AAPL: 100, GOOGL: 20, MSFT: 50 to index2?',
                    'Optimize my holdings to track the S&P 500'
                ],
            ),
            AgentSkill(
                id='tax_loss_harvesting',
                name='Tax Loss Harvesting',
                description='Optimizes tax efficiency through strategic realization of investment losses while maintaining index allocation',
                tags=['tax optimization', 'tax loss harvesting', 'FIFO', 'index-optimized'],
                examples=[
                    'Perform tax loss harvesting on my AAPL and TSLA positions',
                    'What tax losses can I harvest while maintaining index1 allocation?',
                    'Find tax-loss opportunities in my portfolio'
                ],
            ),
            AgentSkill(
                id='index_analysis',
                name='Index Weights & Analysis',
                description='Provides target weights and composition analysis for investment indices',
                tags=['index analysis', 'target weights', 'portfolio composition'],
                examples=[
                    'What are the target weights for index1?',
                    'Show me the composition of index2',
                    'Compare my portfolio allocation to the Russell 2000'
                ],
            ),
            AgentSkill(
                id='price_lookup',
                name='Stock Price Lookup',
                description='Retrieves current stock prices for portfolio analysis and calculations',
                tags=['stock prices', 'market data', 'price lookup'],
                examples=[
                    'What is the current price of AAPL?',
                    'Get me the price for NVDA',
                    'Look up current prices for my holdings'
                ],
            ),
        ]