#!/usr/bin/env python3
"""
Simple LangGraph ReAct Agent Demo
This script demonstrates how to use LangGraph's built-in ReAct agent with custom tools.
"""

import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
from portfolio_tools import get_portfolio_tools
from langchain_core.callbacks import BaseCallbackHandler

# Load environment variables from .env file
load_dotenv()

from langchain_anthropic import ChatAnthropic  # new package per deprecation notice

class PromptDebugCallback(BaseCallbackHandler):
    """Custom callback to capture and print LLM prompts."""
    
    def on_chat_model_start(self, serialized, messages, **kwargs):
        """Called when the chat model starts with the actual messages."""
        print("\n" + "="*80)
        print("🔍 LLM PROMPT DEBUG - Full message structure:")
        print("="*80)
        print(f"📊 Total message batches: {len(messages)}")
        
        # Process each message batch
        for batch_idx, message_batch in enumerate(messages):
            print(f"\n🔸 Batch {batch_idx + 1} contains {len(message_batch)} messages:")
            
            for i, message in enumerate(message_batch):
                print(f"\n--- Message {i+1} ({type(message).__name__}) ---")
                
                if isinstance(message, SystemMessage):
                    print("🤖 SYSTEM PROMPT:")
                    content = str(message.content)
                    print(content[:1000] + "..." if len(content) > 1000 else content)
                elif isinstance(message, HumanMessage):
                    print("👤 USER PROMPT:")
                    content = str(message.content)
                    print(content[:500] + "..." if len(content) > 500 else content)
                elif isinstance(message, AIMessage):
                    print("🧠 AI MESSAGE:")
                    if hasattr(message, 'tool_calls') and message.tool_calls:
                        print(f"Tool calls: {len(message.tool_calls)}")
                        for tc in message.tool_calls:
                            print(f"  - {tc['name']}: {tc['args']}")
                    if message.content:
                        print(f"Content: {message.content[:200]}...")
                else:
                    print(f"📝 {type(message).__name__}:")
                    if hasattr(message, 'content') and message.content:
                        content_preview = str(message.content)[:200] + "..." if len(str(message.content)) > 200 else str(message.content)
                        print(content_preview)
                    if hasattr(message, 'name'):
                        print(f"Tool: {message.name}")
        
        print("\n" + "="*80 + "\n")
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Alternative method to catch prompts."""
        if prompts:
            print("\n🔍 Alternative prompt capture:")
            for i, prompt in enumerate(prompts):
                print(f"Prompt {i+1}: {prompt[:300]}...")

# Define some sample tools
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        # Only allow basic math operations for safety
        allowed_chars = set('0123456789+-*/(). ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations are allowed"
        
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

@tool
def word_counter(text: str) -> str:
    """Count the number of words in a given text."""
    word_count = len(text.split())
    return f"The text contains {word_count} words."

@tool
def temperature_converter(temp: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius, Fahrenheit, and Kelvin."""
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    # Convert to Celsius first
    if from_unit == "fahrenheit":
        celsius = (temp - 32) * 5/9
    elif from_unit == "kelvin":
        celsius = temp - 273.15
    elif from_unit == "celsius":
        celsius = temp
    else:
        return "Error: Supported units are celsius, fahrenheit, kelvin"
    
    # Convert from Celsius to target unit
    if to_unit == "fahrenheit":
        result = celsius * 9/5 + 32
    elif to_unit == "kelvin":
        result = celsius + 273.15
    elif to_unit == "celsius":
        result = celsius
    else:
        return "Error: Supported units are celsius, fahrenheit, kelvin"
    
    return f"{temp}°{from_unit.title()} is {result:.2f}°{to_unit.title()}"

def create_react_agent_demo():
    """Create and demonstrate the ReAct agent with sample tools."""
    
    # Initialize the LLM with debug callback
    debug_callback = PromptDebugCallback()
    llm_openai = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        callbacks=[debug_callback]
    )


    llm_anthropic = ChatAnthropic(
        #model="claude-3-7-sonnet-20250219",                # Claude model to use
        model = "claude-3-5-haiku-latest",                # Claude model to use
        #model = "claude-sonnet-4-latest",
        temperature=0,                   # deterministic output
        max_retries=2,                   # retry transient errors
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")  # correct named parameter
    )


    llm = llm_anthropic
        
    # Create list of tools - include both demo tools and portfolio tools
    portfolio_tools = get_portfolio_tools()
    tools = [calculator, word_counter, temperature_converter] + portfolio_tools
    
    # Create the ReAct agent using LangGraph's prebuilt function
    agent_executor = create_react_agent(llm, tools, debug = False)
    
    return agent_executor

def run_demo_queries(agent_executor):
    """Run some demo queries to show the agent in action."""
    
    demo_questions = [
        "Show me the target weights for index1",
        # Just one question to clearly see the prompt debugging
    ]

    demo_questions = ['I have a portfolio with {"AAPL": 100, "GOOGL": 20, "MSFT": 50}. How should I rebalance it to match index1 ']


    
    demo_questions = [
        "What's 25 + 30, and then convert that number from Celsius to Fahrenheit?",
    ]

    '''Rebalance below portfolio to matc index1
                      AAPL  100
                      GOOGL 200
                      MSFT  50
                      
                      '''

    demo_questions = ['''Sell 10 % of holdings below
                      AAPL  100
                      GOOGL 200
                      MSFT  50
                      
                      Give me summary of the portfolio before and after.
                      In the summary include ticker, quantity, price, total value
                      Inclde total sum of portflio also
                      
                      '''
    
    ]

    demo_questions = ['''Rebalance below portfolio to match index1
                      AAPL  100
                      
                      
                      
                      ''']
    
    demo_questions = ['''I need tax loss harvesting analysis using FIFO strategy for my portfolio. I have these tax lots:
- AAPL: 50 shares bought on 2023-01-15 at $150 per share
- TSLA: 500 shares bought on 2023-02-10 at $300 per share

Please analyze using lot size of 1 share and limit sales to maximum 50% of portfolio value. Use FIFO strategy (no index targeting).''']
    
    demo_questions_1 = ['''Perform tax loss harvesting analysis for my portfolio using index3 allocation strategy with these positions:
- ABC: 100 shares purchased 2023-01-15 at $80 per share  
- XYZ: 100 shares purchased 2023-02-10 at $120 per share

Use lot size of 1 share with 5% allocation tolerance. Target index3 weights.''']

    print("🤖 LangGraph ReAct Agent Demo\n" + "="*50)
    
    for i, question in enumerate(demo_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        print("-" * 40)
        
        try:
            # Invoke the agent with the question
            response = agent_executor.invoke({"messages": [("user", question), 
                                                          # ("user", "write a quanltative summary of the impact of the action based on company profiles")
                                                           ]})
            #, ("system", "If the user does not provide an index name ask them to provide one.")
            
            # Extract the final answer
            final_message = response["messages"][-1]
            print(f"🎯 Answer: {final_message.content}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 40)

def inspect_agent_graph(agent_executor):
    """Print information about the agent's graph structure."""
    print("\n🔍 Agent Graph Structure:")
    try:
        # For LangGraph's CompiledStateGraph, we can get basic info differently
        print(f"Agent type: {type(agent_executor).__name__}")
        print("Graph successfully compiled and ready to use!")
    except Exception as e:
        print(f"Could not inspect graph structure: {str(e)}")


if __name__ == "__main__":
    try:
        # Create the ReAct agent
        print("🚀 Creating ReAct agent...")
        agent = create_react_agent_demo()
        
        # Inspect the graph structure
        inspect_agent_graph(agent)
        
        # Run demo queries
        run_demo_queries(agent)
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running demo: {str(e)}")
        print("\n💡 Make sure you have set your OPENAI_API_KEY environment variable")


# Optional: Interactive mode
def interactive_mode():
    """Run the agent in interactive mode for custom questions."""
    agent = create_react_agent_demo()
    
    print("\n🎮 Interactive Mode - Ask any question (type 'quit' to exit)")
    print("="*60)
    
    while True:
        question = input("\n❓ Your question: ")
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        try:
            response = agent.invoke({"messages": [("user", question)]})
            final_message = response["messages"][-1]
            print(f"🎯 Answer: {final_message.content}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")

# Uncomment the line below to run in interactive mode instead
# interactive_mode()

