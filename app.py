"""
FastAPI application for Simple LangGraph Agent.
Implements Watsonx Orchestrate external agent API.
"""

import logging
import os
import time
import uuid

from agent import run_agent
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    MessageResponse,
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Simple LangGraph Agent API",
    description="A simple external agent for Watsonx Orchestrate with calculator and greeting tools",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Simple LangGraph Agent API is running",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Chat completions endpoint compatible with Watsonx Orchestrate.

    This endpoint receives messages in Watsonx Orchestrate format,
    runs them through the LangGraph agent, and returns a response.
    """
    logger.info(f"Received chat completion request: {request.model_dump_json()}")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable not set",
        )

    # Handle streaming requests by returning non-streaming response
    # watsonx Orchestrate may request streaming, but we'll return complete response
    if request.stream:
        logger.info("Streaming requested but not supported, returning complete response")
        # Continue with non-streaming response

    # # Check if streaming is requested
    # if request.stream:
    #     raise HTTPException(
    #         status_code=501,
    #         detail="Streaming is not yet implemented. Please set stream=false.",
    #     )

    try:
        # Convert messages to dict format for agent
        messages = [msg.model_dump() for msg in request.messages]

        # Run the agent
        logger.info(f"Running agent with {len(messages)} messages")
        response_content = run_agent(messages)
        logger.info(f"Agent response: {response_content}")

        # Build response
        response = ChatCompletionResponse(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model=request.model or "gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=MessageResponse(role="assistant", content=response_content),
                    finish_reason="stop",
                )
            ],
        )

        return JSONResponse(content=response.model_dump())

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    # Check for API key before starting
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set!")
        logger.error("Please set it in your .env file or environment")
        exit(1)

    logger.info("Starting Simple LangGraph Agent API server...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
