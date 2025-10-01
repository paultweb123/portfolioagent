"""
Agent Configuration Protocol

This module defines the abstract interface that all domain-specific agent 
configurations must implement to work with the generic server framework.
"""

from typing import Protocol, List
from langchain_core.tools import BaseTool
from a2a.types import AgentCard, AgentSkill
import os

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
        """Validate domain-specific environment variables and dependencies"""
        # Get model source (default to 'anthropic' if not specified)
        model_source = os.getenv('model_source', 'google').lower()

        print(f"Validating environment for model source: {model_source}")
        
        # Check for required API keys based on model source
        if model_source in('google', 'gemini'):
            if not os.getenv('GOOGLE_API_KEY') and not os.getenv('GEMINI_API_KEY'):
                raise ValueError("Either GOOGLE_API_KEY or GEMINI_API_KEY environment variable required for Google model source")
            print("✅ Found Google API key")            
        elif model_source == 'anthropic':
            if not os.getenv('ANTHROPIC_API_KEY'):
                raise ValueError("ANTHROPIC_API_KEY environment variable required for Anthropic model source")
            print("✅ Found Anthropic API key")
        elif model_source == 'openai':
            if not os.getenv('OPENAI_API_KEY'):
                raise ValueError("OPENAI_API_KEY environment variable required for OpenAI model source")
            print("✅ Found OpenAI API key")            
        else:
            raise ValueError(f"Unsupported model source '{model_source}'. Must be 'google', 'anthropic', or 'openai'")  
        
    
    @property
    def domain_name(self) -> str:
        """
        Return the domain name (e.g., 'finance', 'portfolio').
        
        Returns:
            String identifier for this domain
        """
        ...