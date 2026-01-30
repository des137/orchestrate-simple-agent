"""
Pydantic models for Watsonx Orchestrate external agent API.
Defines the request/response format for chat completions endpoint.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A message in the conversation."""

    role: str = Field(
        ...,
        description="The role of the message sender",
        pattern="^(user|assistant|system|tool)$",
    )
    content: Optional[str] = Field(
        None, description="The content of the message. Can be null."
    )
    tool_call_id: Optional[str] = Field(
        None, description="Tool call ID if role is tool."
    )


class ChatCompletionRequest(BaseModel):
    """Request format for chat completions."""

    model: Optional[str] = Field(
        default="gpt-4o-mini", description="ID of the model to use"
    )
    messages: List[Message] = Field(..., description="List of messages in conversation")
    stream: Optional[bool] = Field(
        False, description="Whether to stream responses (not yet implemented)"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Contextual information for the request"
    )


class MessageResponse(BaseModel):
    """Response message format."""

    role: str = Field(..., description="The role of the message sender")
    content: str = Field(..., description="The content of the message")


class Choice(BaseModel):
    """A choice in the completion response."""

    index: int = Field(..., description="The index of the choice")
    message: MessageResponse = Field(..., description="The message")
    finish_reason: Optional[str] = Field(
        None, description="The reason the message generation finished"
    )


class ChatCompletionResponse(BaseModel):
    """Response format for chat completions."""

    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="The type of object returned")
    created: int = Field(..., description="Timestamp of when completion was created")
    model: str = Field(..., description="The model used for generating completion")
    choices: List[Choice] = Field(..., description="List of completion choices")
