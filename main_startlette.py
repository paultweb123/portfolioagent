#!/usr/bin/env python3
"""
Starlette Web App for LangGraph ReAct Agent Demo
This web application provides an interface to interact with the ReAct agent through a web browser.
"""

import os
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import HTMLResponse, JSONResponse
from starlette.requests import Request
from starlette.templating import Jinja2Templates
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

# Import the agent creation logic from main.py
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from portfolio_tools import get_portfolio_tools
from portfolio_tools_pydantic import pydantic_rebalance_portfolio
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Templates setup
templates = Jinja2Templates(directory="templates")

# Global agent variable (will be initialized on startup)
agent_executor = None

# Define the same tools from main.py
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

def create_react_agent_instance():
    """Create and return the ReAct agent instance."""
    
    # Initialize the LLM with required parameters
    llm_anthropic = ChatAnthropic(
        model_name="claude-3-5-haiku-20241022",
        timeout=60,
        stop=None
    )
    
    # Create list of tools
    portfolio_tools = get_portfolio_tools()
    tools = [calculator, word_counter, temperature_converter] + portfolio_tools + [pydantic_rebalance_portfolio]
    
    # Create the ReAct agent
    agent_executor = create_react_agent(llm_anthropic, tools, debug=False)
    
    return agent_executor

async def homepage(request: Request):
    """Serve the main page with input/output form."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "default_text": "Rebalance below portfolio to match index1\n                      AAPL  100"
    })

async def process_query(request: Request):
    """Process the user query through the ReAct agent."""
    try:
        # Get form data
        form_data = await request.form()
        user_input = form_data.get("user_input", "")
        
        # Convert to string and strip whitespace
        user_input = str(user_input).strip()
        
        if not user_input:
            return JSONResponse({
                "error": "Please provide some input text.",
                "output": ""
            })
        
        # Ensure agent is available
        if agent_executor is None:
            return JSONResponse({
                "error": "Agent not initialized. Please try again.",
                "output": ""
            })
        
        # Process with the ReAct agent
        response = agent_executor.invoke({"messages": [("user", user_input)]})
        
        # Extract the final answer
        final_message = response["messages"][-1]
        output_text = final_message.content
        
        return JSONResponse({
            "success": True,
            "output": output_text
        })
        
    except Exception as e:
        return JSONResponse({
            "error": f"Error processing query: {str(e)}",
            "output": ""
        })

# Define routes
routes = [
    Route("/", homepage),
    Route("/process", process_query, methods=["POST"]),
]

# Middleware
middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

# Create the Starlette application
app = Starlette(routes=routes, middleware=middleware)

@app.on_event("startup")
async def startup_event():
    """Initialize the agent on startup."""
    global agent_executor
    print("🚀 Initializing ReAct agent...")
    agent_executor = create_react_agent_instance()
    print("✅ Agent initialized successfully!")

if __name__ == "__main__":
    # Run the application
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")