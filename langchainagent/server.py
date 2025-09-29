import logging
import os
import sys

import click
import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    BasePushNotificationSender,
    InMemoryPushNotificationConfigStore,
    InMemoryTaskStore,
)
from a2a.types import AgentCapabilities
    
from dotenv import load_dotenv

from langchainagent.agent import LangchainReactAgent
from langchainagent.agent_executor import LangchainReactAgentExecutor
from langchainagent.agent_config import AgentConfiguration


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def create_agent_server(config: AgentConfiguration, host: str = 'localhost', port: int = 10000):
    """
    Generic server creation function that works with any domain configuration.
    
    Args:
        config: AgentConfiguration instance for the specific domain
        host: Server host address
        port: Server port number
    """
    try:
        # Validate domain-specific environment
        config.validate_environment()
        
        # Get agent card and tools from configuration
        agent_card = config.get_agent_card(host, port)
        tools = config.get_tools()
        
        # Create server components
        capabilities = AgentCapabilities(streaming=True, push_notifications=True)
        
        # AgentCard from config already has the necessary details
        # Just ensure it has the correct URL
        if hasattr(agent_card, 'url'):
            agent_card.url = f'http://{host}:{port}/'

        # Create server infrastructure
        httpx_client = httpx.AsyncClient()
        push_config_store = InMemoryPushNotificationConfigStore()
        push_sender = BasePushNotificationSender(
            httpx_client=httpx_client,
            config_store=push_config_store
        )
        agent = LangchainReactAgent(tools=tools)
        request_handler = DefaultRequestHandler(
            agent_executor=LangchainReactAgentExecutor(agent=agent),
            task_store=InMemoryTaskStore(),
            push_config_store=push_config_store,
            push_sender=push_sender
        )
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        print(f"🚀 Starting {config.domain_name} agent server on {host}:{port}")
        print(f"✅ {agent_card.name} v{agent_card.version} ready")
        print(f"📊 Available skills: {len(agent_card.skills)} loaded")
        print(f"🌐 Server running at http://{host}:{port}")

        uvicorn.run(server.build(), host=host, port=port)

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        sys.exit(1)
    except Exception as e:
        logger.error(f'An error occurred during server startup: {e}')
        sys.exit(1)




def run_agent_server(config_class, agent_description: str = "Agent", emoji: str = "🤖"):
    """
    Generic CLI server runner for any domain configuration.
    
    This function provides a complete CLI interface with logging, validation,
    startup banners, and error handling for any agent configuration.
    
    Args:
        config_class: AgentConfiguration class (not instance) for the domain
        agent_description: Description for CLI help and startup banner
        emoji: Emoji to display in startup banner
    
    Usage:
        from langchainagent.server import run_agent_server
        from your_domain.config import YourAgentConfig
        
        run_agent_server(YourAgentConfig, "Your Domain Agent", "🎯")
    """
    @click.command()
    @click.option('--host', default='localhost', help='Server host address')
    @click.option('--port', default=10000, type=int, help='Server port number')
    @click.option('--verbose', is_flag=True, help='Enable verbose logging')
    def main(host: str, port: int, verbose: bool):
        f"""Launch {agent_description} Server"""
        
        # Setup logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
            print("🔧 Verbose logging enabled")
        else:
            logging.basicConfig(level=logging.INFO)
        
        try:
            print(f"{emoji} {agent_description} Server")
            print("=" * 50)
            
            # Create domain configuration
            config = config_class()
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
    
    # Execute the CLI
    main()
