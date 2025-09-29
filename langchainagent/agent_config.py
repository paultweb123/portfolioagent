"""
Agent Configuration Protocol

This module defines the abstract interface that all domain-specific agent 
configurations must implement to work with the generic server framework.
"""

from typing import Protocol, List
from langchain_core.tools import BaseTool
from a2a.types import AgentCard, AgentSkill


class AgentConfiguration(Protocol):
    """
    Protocol that all agent configurations must implement.
    
    This ensures type safety and consistent interfaces across all domain
    implementations while allowing the generic server to work with any
    domain-specific configuration.
    """
    
    def get_agent_card(self, host: str, port: int) -> AgentCard:
        """
        Return the agent card for this domain.
        
        Args:
            host: Server host address
            port: Server port number
            
        Returns:
            AgentCard with domain-specific information
        """
        ...
    
    def get_tools(self) -> List[BaseTool]:
        """
        Return the LangChain tools for this domain.
        
        Returns:
            List of LangChain BaseTool instances for this domain
        """
        ...
    
    def validate_environment(self) -> None:
        """
        Validate domain-specific environment variables and dependencies.
        
        Raises:
            ValueError: If required environment variables are missing
            ImportError: If required dependencies are not available
        """
        ...
    
    @property
    def domain_name(self) -> str:
        """
        Return the domain name (e.g., 'finance', 'portfolio').
        
        Returns:
            String identifier for this domain
        """
        ...