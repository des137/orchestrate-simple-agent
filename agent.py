"""
Simple LangGraph Agent with calculator and greeting tools.
Refactored for use as a FastAPI service.
"""

from typing import List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


# Define tools
@tool
def calculator(operation: str, a: float, b: float) -> str:
    """Perform basic math operations: add, subtract, multiply, or divide.

    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        a: First number
        b: Second number

    Returns:
        The result of the operation
    """
    if operation == "add":
        return str(a + b)
    elif operation == "subtract":
        return str(a - b)
    elif operation == "multiply":
        return str(a * b)
    elif operation == "divide":
        if b == 0:
            return "Error: Division by zero"
        return str(a / b)
    else:
        return "Error: Unknown operation"


@tool
def get_greeting(name: str = "friend") -> str:
    """Generate a friendly greeting for someone.

    Args:
        name: The name of the person to greet

    Returns:
        A friendly greeting message
    """
    return f"Hello {name}! Nice to meet you! 👋"


# List of available tools
TOOLS = [calculator, get_greeting]


def agent_node(state: MessagesState):
    """Main agent node that decides what to do based on user input."""
    # Initialize OpenAI model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Bind tools to the model
    model_with_tools = model.bind_tools(TOOLS)

    # Add system message for context
    messages = state["messages"]
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [
            SystemMessage(
                content="You are a helpful assistant that can greet people and help with math calculations. "
                "Use the calculator tool for math problems and the get_greeting tool to greet users."
            )
        ] + messages

    # Get response from model
    response = model_with_tools.invoke(messages)

    return {"messages": [response]}


def build_graph():
    """Build and compile the LangGraph workflow."""
    # Create the graph
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(TOOLS))

    # Add edges
    workflow.add_edge(START, "agent")

    # Add conditional edges - if tools are called, go to tools node, otherwise end
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    # After tools are called, go back to agent to generate final response
    workflow.add_edge("tools", "agent")

    # Compile the graph
    return workflow.compile()


def convert_messages_to_langgraph(messages: List[dict]) -> List[BaseMessage]:
    """Convert Watsonx Orchestrate message format to LangGraph format.

    Args:
        messages: List of message dicts with 'role' and 'content'

    Returns:
        List of LangChain BaseMessage objects
    """
    langgraph_messages = []

    for msg in messages:
        role = msg["role"].lower()
        content = msg.get("content", "")

        if role == "user" or role == "human":
            langgraph_messages.append(HumanMessage(content=content))
        elif role == "assistant" or role == "ai":
            langgraph_messages.append(AIMessage(content=content))
        elif role == "system":
            langgraph_messages.append(SystemMessage(content=content))
        elif role == "tool":
            # Tool messages need special handling
            tool_call_id = msg.get("tool_call_id", "")
            langgraph_messages.append(
                ToolMessage(content=content, tool_call_id=tool_call_id)
            )
        else:
            # Default to human message for unknown roles
            langgraph_messages.append(HumanMessage(content=content))

    return langgraph_messages


def run_agent(messages: List[dict]) -> str:
    """Run the agent with the given messages and return the final response.

    Args:
        messages: List of message dicts in Watsonx Orchestrate format

    Returns:
        The final response content as a string
    """
    # Build the graph
    graph = build_graph()

    # Convert messages to LangGraph format
    langgraph_messages = convert_messages_to_langgraph(messages)

    # Run the agent
    response = graph.invoke({"messages": langgraph_messages})

    # Extract the final message content
    final_message = response["messages"][-1]

    if hasattr(final_message, "content"):
        return final_message.content
    else:
        return str(final_message)
