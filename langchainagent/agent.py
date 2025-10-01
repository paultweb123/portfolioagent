import os

from collections.abc import AsyncIterable
from typing import Any, Literal

import httpx

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from langchain_anthropic import ChatAnthropic  

memory = MemorySaver()

class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str


class LangchainReactAgent:
    """An angent that uses Langchain's React framework and uses tools to respond."""

    SYSTEM_INSTRUCTION = (
        "You are a helpful client that answers questions using only the tools available to you. "
        "If the user asks about topics outside those tools' capabilities, politely explain that you can only assist with queries "
        "related to the tools and their functions. "
        "Do not attempt to answer unrelated questions or misuse the tools."
    )

    

    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information to complete the request.'
        'Set response status to error if there is an error while processing the request.'
        'Set response status to completed if the request is complete.'
    )

    def __init__(self, tools):
        model_source = os.getenv('model_source', 'google')
        
        if model_source in('google', 'gemini'):
            print("⭐Initializing Google Gemini model...")
            self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        elif model_source == 'anthropic':
            print("⭐Initializing Anthropic Claude model...")
            self.model =    ChatAnthropic(
                model = "claude-3-5-haiku-latest",                # Claude model to use
                temperature=0,                   # deterministic output
                max_retries=2,                   # retry transient errors
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")  # correct named parameter
            )
        elif model_source == 'openai':
            print("⭐Initializing OpenAI GPT model...")
            self.model = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0,               
            )

        else:
            raise ValueError(f'Unsupported model source: {model_source}')
        
        self.tools = tools

        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )

    async def stream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Gathering information...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing the results...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        if structured_response and isinstance(
            structured_response, ResponseFormat
        ):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain']
