import logging
import os
from typing import Any
from uuid import uuid4

import httpx

from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)

from pprint import pprint, pformat

def extract_and_print_text_parts(data: dict, prefix: str = "📝") -> None:
    """Extract and print text parts from response data in a readable format."""
    
    def extract_text_from_parts(parts_list):
        """Extract text from a list of parts."""
        texts = []
        for part in parts_list:
            if isinstance(part, dict) and part.get('kind') == 'text':
                texts.append(part.get('text', ''))
        return texts
    
    # Handle different response structures
    if 'result' in data:
        result = data['result']
        
        # Handle artifacts in completed responses
        if 'artifacts' in result:
            for artifact in result['artifacts']:
                if 'parts' in artifact:
                    texts = extract_text_from_parts(artifact['parts'])
                    for text in texts:
                        print(f"\n{prefix} AGENT RESPONSE:")
                        print(f"🔵 {text}")
                        print("-" * 50)
        
        # Handle single artifact in streaming responses
        elif 'artifact' in result and 'parts' in result['artifact']:
            texts = extract_text_from_parts(result['artifact']['parts'])
            for text in texts:
                print(f"\n{prefix} STREAMING ARTIFACT:")
                print(f"🟢 {text}")
                print("-" * 50)
        
        # Handle status updates with messages
        elif 'status' in result and 'message' in result['status']:
            message = result['status']['message']
            if 'parts' in message:
                texts = extract_text_from_parts(message['parts'])
                for text in texts:
                    print(f"\n{prefix} STATUS MESSAGE:")
                    print(f"🟡 {text}")
                    print("-" * 30)


async def main() -> None:
    # Configure logging to show INFO level messages
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)  # Get a logger instance

    # --8<-- [start:A2ACardResolver]

    # Allow port to be configured via environment variable
    
    base_url = 'http://localhost:10000'
    logger.info(f"Connecting to agent server at: {base_url}")

    # Configure timeout settings
    timeout_config = httpx.Timeout(
        connect=10.0,  # 10 seconds to connect
        read=60.0,     # 60 seconds to read response (portfolio calculations can be slow)
        write=10.0,    # 10 seconds to send request
        pool=10.0      # 10 seconds to get connection from pool
    )

    async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
            # agent_card_path uses default, extended_agent_card_path also uses default
        )
        # --8<-- [end:A2ACardResolver]

        # Fetch Public Agent Card and Initialize Client
        final_agent_card_to_use: AgentCard | None = None

        try:
            logger.info(
                f'Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}'
            )
            _public_card = (
                await resolver.get_agent_card()
            )  # Fetches from default public path
            logger.info('Successfully fetched public agent card:')
            logger.info(
                _public_card.model_dump_json(indent=2, exclude_none=True)
            )
            final_agent_card_to_use = _public_card
            logger.info(
                '\nUsing PUBLIC agent card for client initialization (default).'
            )

            if _public_card.supports_authenticated_extended_card:
                try:
                    logger.info(
                        '\nPublic card supports authenticated extended card. '
                        'Attempting to fetch from: '
                        f'{base_url}{EXTENDED_AGENT_CARD_PATH}'
                    )
                    auth_headers_dict = {
                        'Authorization': 'Bearer dummy-token-for-extended-card'
                    }
                    _extended_card = await resolver.get_agent_card(
                        relative_card_path=EXTENDED_AGENT_CARD_PATH,
                        http_kwargs={'headers': auth_headers_dict},
                    )
                    logger.info(
                        'Successfully fetched authenticated extended agent card:'
                    )
                    logger.info(
                        _extended_card.model_dump_json(
                            indent=2, exclude_none=True
                        )
                    )
                    final_agent_card_to_use = (
                        _extended_card  # Update to use the extended card
                    )
                    logger.info(
                        '\nUsing AUTHENTICATED EXTENDED agent card for client '
                        'initialization.'
                    )
                except Exception as e_extended:
                    logger.warning(
                        f'Failed to fetch extended agent card: {e_extended}. '
                        'Will proceed with public card.',
                        exc_info=True,
                    )
            elif (
                _public_card
            ):  # supports_authenticated_extended_card is False or None
                logger.info(
                    '\nPublic card does not indicate support for an extended card. Using public card.'
                )

        except Exception as e:
            logger.error(
                f'Critical error fetching public agent card: {e}', exc_info=True
            )
            raise RuntimeError(
                'Failed to fetch the public agent card. Cannot continue.'
            ) from e

        print("\n\n⭐⭐⭐⭐ Send Message Test ⭐⭐⭐⭐\n\n")
        # --8<-- [start:send_message]
        try:
            # Use A2AClient with improved timeout handling
            client = A2AClient(
                httpx_client=httpx_client, agent_card=final_agent_card_to_use
            )
            logger.info('A2AClient initialized with extended timeouts.')

            send_message_payload: dict[str, Any] = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': '''Rebalance below portfolio to match index1
                          AAPL  100'''}
                    ],
                    'message_id': uuid4().hex,
                },
            }
            request = SendMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )

            logger.info('Sending message to agent...')
            response = await client.send_message(request)
            logger.info('Received response from agent.')
            
            response_data = response.model_dump(mode='json', exclude_none=True)
            logging.debug(  pformat(response_data) )
            
            # Extract and display readable text parts
            extract_and_print_text_parts(response_data, "💎")
            # --8<-- [end:send_message]

            logger.info('Testing streaming message...')


            print("\n\n⭐⭐⭐⭐ Streaming Test ⭐⭐⭐⭐\n\n")
            


        
        
        
            # --8<-- [start:send_message_streaming]
            streaming_request = SendStreamingMessageRequest(
                id=str(uuid4()), params=MessageSendParams(**send_message_payload)
            )

            stream_response = client.send_message_streaming(streaming_request)
            i = 0
            async for chunk in stream_response:
                logger.info(f"\n✅Received chunk {i}")
                response_data = chunk.model_dump(mode='json', exclude_none=True)
                logging.debug(  pformat(response_data) )
                extract_and_print_text_parts(response_data, "💎")
                i += 1
            # --8<-- [end:send_message_streaming]
            
        except Exception as e:
            logger.error(f"Error during message exchange: {e}", exc_info=True)
            raise


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
