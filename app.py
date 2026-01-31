"""
FastAPI application for Simple LangGraph Agent.
Implements Watsonx Orchestrate external agent API.
"""

import asyncio
import json
import logging
import os
import time
import uuid

from agent import run_agent
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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
    Supports both streaming and non-streaming responses.
    """
    logger.info(f"Received chat completion request: {request.model_dump_json()}")

    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable not set",
        )

    try:
        # Convert messages to dict format for agent
        messages = [msg.model_dump() for msg in request.messages]

        if request.stream:
            # Streaming response
            logger.info(f"Running agent with {len(messages)} messages (streaming)")
            
            async def generate():
                try:
                    # Run agent and get response
                    response_content = run_agent(messages)
                    logger.info(f"Agent response (streaming): {response_content}")
                    
                    # Send response as streaming chunks
                    # Split into words for streaming effect
                    words = response_content.split()
                    
                    for i, word in enumerate(words):
                        chunk = {
                            "id": str(uuid.uuid4()),
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": request.model or "gpt-4o-mini",
                            "choices": [{
                                "index": 0,
                                "delta": {
                                    "content": word + (" " if i < len(words) - 1 else "")
                                },
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        await asyncio.sleep(0.01)  # Small delay for streaming effect
                    
                    # Send final chunk with finish_reason
                    final_chunk = {
                        "id": str(uuid.uuid4()),
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model or "gpt-4o-mini",
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    logger.error(f"Error in streaming: {str(e)}", exc_info=True)
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "internal_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"
                }
            )
        else:
            # Non-streaming response (original code)
            logger.info(f"Running agent with {len(messages)} messages")
            response_content = run_agent(messages)
            logger.info(f"Agent response: {response_content}")

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
