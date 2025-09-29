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
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
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

