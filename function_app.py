# Import necessary libraries
import logging
import json
import os
import io
from typing import List, Optional, Dict, Any, Callable

import azure.functions as func # Core Azure Functions library

# LangChain and LangGraph imports
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import START, StateGraph, MessagesState, END
from langgraph.prebuilt import tools_condition, ToolNode
from openai import APIConnectionError, AuthenticationError, RateLimitError

# --- Basic Logging Setup ---
# Azure Functions integrates with its own logging. Get the logger for the function module.
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Logs will be associated with this module/file
logger.setLevel(logging.INFO)
# You might not need basicConfig if Azure Functions handles handler setup.

# --- Azure Functions App Initialization ---
# Initialize the FunctionApp instance. http_auth_level controls access.
# Common levels: FUNCTION (requires function key), ANONYMOUS (no key), ADMIN (master key)
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)
logger.info("Azure FunctionApp initialized.")

# ==============================================================================
# --- Tool Definitions ---
# Define or import your agent's tools here.
# ==============================================================================

# Example Tool 1: Add
def add(a: float, b: float) -> float | str:
    """Adds two numbers and returns the result."""
    logger.info(f"--- Calling add tool with: a={a}, b={b} ---")
    try:
        return float(a) + float(b)
    except (TypeError, ValueError) as e:
        logger.error(f"Error in add tool: {e}")
        return f"Error: Invalid input for addition - {e}"

# Example Tool 2: Multiply
def multiply(a: float, b: float) -> float | str:
    """Multiplies two numbers and returns the result."""
    logger.info(f"--- Calling multiply tool with: a={a}, b={b} ---")
    try:
        return float(a) * float(b)
    except (TypeError, ValueError) as e:
        logger.error(f"Error in multiply tool: {e}")
        return f"Error: Invalid input for multiplication - {e}"

# Example Tool 3: Divide
def divide(a: float, b: float) -> float | str:
    """Divides two numbers and returns the result."""
    logger.info(f"--- Calling divide tool with: a={a}, b={b} ---")
    try:
        num_a = float(a)
        num_b = float(b)
        if num_b == 0:
            logger.warning("Division by zero attempted.")
            return "Error: Cannot divide by zero."
        return num_a / num_b
    except (TypeError, ValueError) as e:
        logger.error(f"Error in divide tool: {e}")
        return f"Error: Invalid input for division - {e}"

# --- Add your new tool functions above this line ---


# ==============================================================================
# --- Tool Registration ---
# ==============================================================================
agent_tools: List[Callable | BaseTool] = [
    add,
    multiply,
    divide,
    # --- Add your new tool functions here ---
]


# ==============================================================================
# --- Agent Initialization (Singleton Pattern for Azure Functions) ---
# Use global variables to store the initialized agent components.
# ==============================================================================
graph = None
is_agent_initialized = False # Flag to track initialization

def initialize_agent():
    """Initializes the LLM, binds tools, defines system message, and compiles the LangGraph agent."""
    global graph, is_agent_initialized
    if is_agent_initialized:
        logger.info("Agent already initialized in this worker instance.")
        return

    logger.info("Initializing agent for the first time in this worker instance...")

    # --- Azure OpenAI Configuration ---
    try:
        api_key = os.environ["AZURE_OPENAI_API_KEY"]
        azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        api_version = os.environ["OPENAI_API_VERSION"]
        deployment_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
        logger.info("Azure environment variables loaded successfully.")
    except KeyError as e:
        error_msg = f"Missing Azure OpenAI environment variable: {e}. Set in App Settings or local.settings.json."
        logger.error(error_msg)
        raise ValueError(error_msg) from e # Fail fast if config missing

    # Initialize LLM
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version,
            azure_deployment=deployment_name, temperature=0, max_retries=2,
        )
        logger.info("AzureChatOpenAI initialized.")
    except Exception as e:
        logger.error(f"Fatal Error initializing AzureChatOpenAI: {e}", exc_info=True)
        raise RuntimeError(f"Could not initialize AzureChatOpenAI: {e}") from e

    # Bind Tools
    if not agent_tools:
        logger.warning("No tools registered. Binding base LLM.")
        llm_with_tools = llm
    else:
        try:
            llm_with_tools = llm.bind_tools(agent_tools)
            logger.info(f"Successfully bound {len(agent_tools)} tools.")
        except Exception as e:
            logger.error(f"Fatal Error binding tools: {e}", exc_info=True)
            raise RuntimeError(f"Could not bind tools: {e}") from e

    # System Message
    sys_msg_content = (
        "You are a helpful assistant using Azure Functions. Use provided tools. "
        "Current tools: add, multiply, divide. " # <-- UPDATE THIS
        "Respond clearly."
    )
    sys_msg = SystemMessage(content=sys_msg_content)
    logger.info("System message defined.")

    # --- Agent Logic Node ---
    def assistant(state: MessagesState):
        logger.info("--- Calling Assistant Node ---")
        messages = state["messages"]
        if not messages: return {"messages": [AIMessage(content="Hello! How can I help?")]}

        input_log = [f"{m.type}: {m.content[:50]}..." for m in messages] # Log snippet
        logger.info(f"Assistant input messages: {input_log}")

        try:
            response: BaseMessage = llm_with_tools.invoke([sys_msg] + messages)
            logger.info("LLM invoked.")
            if hasattr(response, 'content'): logger.info(f"Assistant response content snippet: '{response.content[:50]}...'")
            if hasattr(response, 'tool_calls') and response.tool_calls: logger.debug(f"LLM tool calls: {response.tool_calls}")
            return {"messages": [response]}
        except (APIConnectionError, AuthenticationError, RateLimitError) as e:
            error_type = type(e).__name__
            logger.error(f"Azure API Error ({error_type}) in Assistant: {e}")
            return {"messages": messages + [AIMessage(content=f"Error: AI service unavailable ({error_type}).")]}
        except Exception as e:
            logger.error(f"Unexpected Error in Assistant: {e}", exc_info=True)
            return {"messages": messages + [AIMessage(content=f"Error: Assistant error. Details: {e}")]}

    # --- Graph Definition ---
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    tool_node = ToolNode(agent_tools)
    builder.add_node("tools", tool_node)
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "assistant")

    # Compile and store the graph globally
    try:
        graph = builder.compile() # Assign to global 'graph'
        is_agent_initialized = True # Set flag
        logger.info("LangGraph agent compiled and initialized successfully.")
    except Exception as e:
        logger.error(f"Fatal Error compiling LangGraph: {e}", exc_info=True)
        raise RuntimeError(f"Could not compile LangGraph: {e}") from e

# ==============================================================================
# --- Azure Function HTTP Trigger ---
# ==============================================================================

@app.route(route="BasicMathAIAgentLangChain", methods=["POST"]) # Define route and allow POST
def BasicMathAIAgentLangChain(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function HTTP Trigger entry point using V2 model.
    Handles POST requests to the defined route.
    """
    global graph # Access the globally initialized graph

    # --- Initialization Check ---
    # Ensure the agent is initialized before processing requests.
    try:
        if not is_agent_initialized:
            initialize_agent()
        if graph is None: # Double-check after initialization attempt
             logger.critical("Agent graph is None after initialization attempt.")
             return func.HttpResponse(
                 json.dumps({"error": "Agent initialization failed. Check logs."}),
                 status_code=500, mimetype="application/json"
             )
    except Exception as e:
         logger.critical(f"Failed to initialize agent during request: {e}", exc_info=True)
         return func.HttpResponse(
             json.dumps({"error": f"Agent initialization failed: {e}"}),
             status_code=500, mimetype="application/json"
         )

    logging.info('Python HTTP trigger function processed a request.') # Use standard logging

    # --- Request Body Processing ---
    try:
        req_body = req.get_json()
        message = req_body.get('message')
    except ValueError:
        logger.warning("Request body is not valid JSON.")
        return func.HttpResponse(
             json.dumps({"error": "Request body must be valid JSON."}),
             status_code=400, mimetype="application/json"
        )
    except Exception as e:
        logger.error(f"Error reading request body: {e}")
        return func.HttpResponse(
             json.dumps({"error": "Could not process request body."}),
             status_code=400, mimetype="application/json"
        )


    if not message:
        logger.warning("Request JSON missing 'message' key.")
        return func.HttpResponse(
             json.dumps({"error": "Please include 'message' in the JSON request body"}),
             status_code=400, mimetype="application/json"
        )

    logger.info(f"Received message: '{message[:100]}...'") # Log snippet

    # --- Log Capture Setup ---
    log_stream = io.StringIO()
    log_handler = logging.StreamHandler(log_stream)
    log_handler.setFormatter(log_formatter)
    # Add handler to the current module's logger to capture relevant logs
    current_logger = logging.getLogger(__name__)
    current_logger.addHandler(log_handler)
    # -------------------------

    log_contents = []
    route_nodes = []
    response_content = "Error: Processing failed."
    last_event_data = None
    status_code = 500 # Default to server error

    # --- Agent Invocation ---
    try:
        logger.info("Streaming graph execution...")
        graph_input = {"messages": [HumanMessage(content=message)]}

        # Use graph.stream()
        for event in graph.stream(graph_input):
            node_name = list(event.keys())[0]
            route_nodes.append(node_name)
            logger.debug(f"Graph entered node: {node_name}")
            last_event_data = event[node_name]

        logger.info(f"Graph streaming complete. Route: {route_nodes}")

        # Extract final response
        if last_event_data and isinstance(last_event_data, dict) and 'messages' in last_event_data:
            final_messages = last_event_data['messages']
            if final_messages:
                last_message = final_messages[-1]
                if isinstance(last_message, AIMessage):
                    response_content = last_message.content
                    logger.info(f"Final agent response extracted.")
                    status_code = 200 # Success
                else: # Find last AIMessage if needed
                    logger.warning(f"Last message not AIMessage: {last_message.type}")
                    for msg in reversed(final_messages):
                        if isinstance(msg, AIMessage):
                             response_content = msg.content; status_code = 200; break
                    else: response_content = "Agent finished without AI response."; status_code = 200 # Still OK technically
            else: response_content = "Error: Agent returned empty message list."
        else: response_content = "Error: Agent stream ended unexpectedly."

    except (APIConnectionError, AuthenticationError, RateLimitError) as e:
        error_type = type(e).__name__
        logger.error(f"Azure API Error ({error_type}) during graph streaming: {e}")
        response_content = f"Error: AI service unavailable ({error_type})."
        status_code = 503 # Service Unavailable
    except Exception as e:
        logger.error(f"Unexpected error during graph streaming: {e}", exc_info=True)
        response_content = f"Error: Internal server error. Check logs for details."
        status_code = 500
    finally:
        # --- Log Capture Cleanup ---
        current_logger.removeHandler(log_handler)
        log_handler.flush()
        log_stream.seek(0)
        log_contents = log_stream.read().splitlines()
        log_stream.close()
        # -------------------------
        logger.info(f"Captured {len(log_contents)} log lines for the request.")

    # --- Prepare Response ---
    response_body = {
        "response": response_content,
        "logs": log_contents,
        "route": route_nodes
    }

    # Return Azure Functions HTTP Response
    return func.HttpResponse(
        json.dumps(response_body),
        status_code=status_code, # Use determined status code
        mimetype="application/json"
    )

# --- How to Use ---
# 1. Save this code as the main file for your function (e.g., function_app.py or __init__.py depending on project structure).
# 2. Define/import tools and add them to the `agent_tools` list.
# 3. Update the `sys_msg_content` to match your agent's tools.
# 4. Create a `requirements.txt` file listing dependencies (azure-functions, langchain-openai, etc.).
# 5. Create `local.settings.json` for local development environment variables (AZURE_OPENAI_*).
# 6. Deploy to Azure Functions, ensuring Application Settings are configured for AZURE_OPENAI_* variables.
# 7. The function will be available at `<your_function_app_url>/api/BasicMathAIAgentLangChain` (or similar, check Azure portal).

