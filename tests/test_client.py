import asyncio
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import boto3
import pytest
from autogen_core import CancellationToken, FunctionCall, Image
from autogen_core.models import (
    AssistantMessage,
    CreateResult,
    FunctionExecutionResultMessage,
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema

from autogen_ext_bedrock._client import BedrockChatCompletionClient
from autogen_ext_bedrock._config import BedrockClientConfigurationConfigModel


@pytest.fixture
def mock_boto3_client():
    """Mock boto3 client for testing."""
    with patch("boto3.client") as mock_client:
        # Mock bedrock client
        mock_bedrock = MagicMock()
        mock_bedrock.list_inference_profiles.return_value = {
            "ResponseMetadata": {"RequestId": "4cb71dd2-4e3c-4eb9-xxxx-88e278d18551", "HTTPStatusCode": 200},
            "inferenceProfileSummaries": [
                {
                    "inferenceProfileName": "US Anthropic Claude 3 Sonnet",
                    "description": "Routes requests to Anthropic Claude 3 Sonnet in us-east-1 and us-west-2.",
                    "inferenceProfileArn": "arn:aws:bedrock:us-east-1:xxxx:inference-profile/us.anthropic.claude-3-sonnet-20240229-v1:0",
                    "models": [
                        {
                            "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
                        },
                        {
                            "modelArn": "arn:aws:bedrock:us-west-2::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0"
                        },
                    ],
                    "inferenceProfileId": "us.anthropic.claude-3-sonnet-20240229-v1:0",
                    "status": "ACTIVE",
                    "type": "SYSTEM_DEFINED",
                }
            ],
        }
        mock_bedrock.list_foundation_models.return_value = {
            "ResponseMetadata": {"RequestId": "9aa58fe5-cdf1-46a1-xxxx-6d84bb86a8d8", "HTTPStatusCode": 200},
            "modelSummaries": [
                {
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-sonnet-20240229-v1:0",
                    "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
                    "modelName": "Claude 3 Sonnet",
                    "providerName": "Anthropic",
                    "inputModalities": ["TEXT", "IMAGE"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "customizationsSupported": [],
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "LEGACY"},
                },
                {
                    "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-haiku-20240307-v1:0",
                    "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
                    "modelName": "Claude 3 Haiku",
                    "providerName": "Anthropic",
                    "inputModalities": ["TEXT"],
                    "outputModalities": ["TEXT"],
                    "responseStreamingSupported": True,
                    "customizationsSupported": [],
                    "inferenceTypesSupported": ["ON_DEMAND"],
                    "modelLifecycle": {"status": "ACTIVE"},
                },
            ],
        }

        # Mock bedrock runtime client
        mock_runtime = MagicMock()
        mock_runtime.converse.return_value = {
            "output": {"message": {"content": [{"text": "This is a test response"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        # Mock streaming response
        mock_stream_response = {
            "stream": [
                {"contentBlockDelta": {"delta": {"text": "This "}}},
                {"contentBlockDelta": {"delta": {"text": "is "}}},
                {"contentBlockDelta": {"delta": {"text": "a test "}}},
                {"contentBlockDelta": {"delta": {"text": "response"}}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
            ]
        }
        mock_runtime.converse_stream.return_value = mock_stream_response

        # Configure boto3.client to return our mocks
        def side_effect(service_name, **kwargs):
            if service_name == "bedrock":
                return mock_bedrock
            elif service_name == "bedrock-runtime":
                return mock_runtime
            return MagicMock()

        mock_client.side_effect = side_effect
        yield mock_client


@pytest.fixture
def client(mock_boto3_client):
    """Create a BedrockChatCompletionClient instance for testing."""
    return BedrockChatCompletionClient(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
    )


class TestBedrockChatCompletionClient:
    """Test cases for BedrockChatCompletionClient."""

    def test_init(self, client, mock_boto3_client):
        """Test client initialization."""
        assert client.model_id == "anthropic.claude-3-sonnet-20240229-v1:0"
        assert client.temperature == 0.7
        assert client.max_tokens == 1000
        assert client.top_p == 0.9
        assert client.region == "us-east-1"  # From env

        # Verify model info was set correctly
        assert client.model_info["vision"] is True
        assert client.model_info["function_calling"] is True
        assert client.model_info["json_output"] is True
        assert client.model_info["family"] == "Bedrock"

    def test_list_models(self, client):
        """Test _list_models method."""
        model_list = client._list_models()
        assert "anthropic.claude-3-sonnet-20240229-v1:0" in model_list
        assert "us.anthropic.claude-3-sonnet-20240229-v1:0" in model_list
        assert model_list["anthropic.claude-3-sonnet-20240229-v1:0"]["modalities"] == ["TEXT", "IMAGE"]
        assert model_list["us.anthropic.claude-3-sonnet-20240229-v1:0"]["modalities"] == ["TEXT", "IMAGE"]

        assert "anthropic.claude-3-haiku-20240307-v1:0" in model_list
        assert "us.anthropic.claude-3-haiku-20240307-v1:0" not in model_list
        assert model_list["anthropic.claude-3-haiku-20240307-v1:0"]["modalities"] == ["TEXT"]

    def test_convert_messages_to_bedrock_format(self, client):
        """Test _convert_messages_to_bedrock_format method."""
        messages = [
            SystemMessage(content="You are a helpful assistant.", type="SystemMessage"),
            UserMessage(content="Hello, how are you?", source="user", type="UserMessage"),
            AssistantMessage(content="I'm doing well, thank you for asking!", source="agent", type="AssistantMessage"),
        ]

        system_prompt, bedrock_messages = client._convert_messages_to_bedrock_format(messages)

        # Check system prompt
        assert system_prompt == "You are a helpful assistant."
        # Check bedrock messages
        assert len(bedrock_messages) == 2
        assert bedrock_messages[0]["role"] == "user"
        assert bedrock_messages[0]["content"][0]["text"] == "Hello, how are you?"
        assert bedrock_messages[1]["role"] == "assistant"
        assert bedrock_messages[1]["content"][0]["text"] == "I'm doing well, thank you for asking!"

    def test_convert_messages_with_function_calls(self, client):
        """Test _convert_messages_to_bedrock_format with function calls."""
        function_call = FunctionCall(
            id="call_123", name="get_weather", arguments='{"location": "San Francisco", "unit": "celsius"}'
        )

        messages = [
            SystemMessage(content="You are a helpful assistant.", type="SystemMessage"),
            UserMessage(content="What's the weather in San Francisco?", source="user", type="UserMessage"),
            AssistantMessage(content=[function_call], source="agent", type="AssistantMessage"),
        ]

        system_prompt, bedrock_messages = client._convert_messages_to_bedrock_format(messages)

        # Check bedrock messages with function calls
        assert len(bedrock_messages) == 2
        assert bedrock_messages[1]["role"] == "assistant"
        assert "toolUse" in bedrock_messages[1]["content"][0]
        assert bedrock_messages[1]["content"][0]["toolUse"]["toolUseId"] == "call_123"
        assert bedrock_messages[1]["content"][0]["toolUse"]["name"] == "get_weather"
        assert bedrock_messages[1]["content"][0]["toolUse"]["input"] == {"location": "San Francisco", "unit": "celsius"}

    def test_prepare_tool_config(self, client):
        """Test _prepare_tool_config method."""
        tool_schema = ToolSchema(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        )

        tool_config = client._prepare_tool_config([tool_schema])

        assert len(tool_config["tools"]) == 1
        assert tool_config["tools"][0]["toolSpec"]["name"] == "get_weather"
        assert tool_config["tools"][0]["toolSpec"]["description"] == "Get the current weather for a location"
        assert "json" in tool_config["tools"][0]["toolSpec"]["inputSchema"]

    def test_extract_usage_from_metadata(self, client):
        """Test _extract_usage_from_metadata method."""
        usage_data = {"inputTokens": 100, "outputTokens": 50}

        usage = client._extract_usage_from_metadata(usage_data)

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50

    def test_process_bedrock_response_text(self, client):
        """Test _process_bedrock_response with text response."""
        response = {
            "output": {"message": {"content": [{"text": "This is a test response"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        messages = [UserMessage(content="Test message", source="user", type="UserMessage")]
        tools = []

        result = client._process_bedrock_response(response, messages, tools, None)

        assert isinstance(result, CreateResult)
        assert result.content == "This is a test response"
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5
        assert result.cached is False

    def test_process_bedrock_response_tool_calls(self, client):
        """Test _process_bedrock_response with tool calls."""
        response = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call_123",
                                "name": "get_weather",
                                "input": {"location": "San Francisco", "unit": "celsius"},
                            }
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        messages = [UserMessage(content="What's the weather in San Francisco?", source="user", type="UserMessage")]
        tools = []

        result = client._process_bedrock_response(response, messages, tools, None)

        assert isinstance(result, CreateResult)
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], FunctionCall)
        assert result.content[0].id == "call_123"
        assert result.content[0].name == "get_weather"
        assert json.loads(result.content[0].arguments) == {"location": "San Francisco", "unit": "celsius"}
        assert result.finish_reason == "function_calls"

    @pytest.mark.asyncio
    async def test_create(self, client):
        """Test create method."""
        messages = [
            SystemMessage(content="You are a helpful assistant.", type="SystemMessage"),
            UserMessage(content="Hello, how are you?", source="user", type="UserMessage"),
        ]

        result = await client.create(messages)

        assert isinstance(result, CreateResult)
        assert result.content == "This is a test response"
        assert result.finish_reason == "stop"
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_create_with_tools(self, client):
        """Test create method with tools."""
        # Mock the converse response to include a tool call
        client.bedrock_runtime.converse.return_value = {
            "output": {
                "message": {
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "call_123",
                                "name": "get_weather",
                                "input": {"location": "San Francisco", "unit": "celsius"},
                            }
                        }
                    ]
                }
            },
            "usage": {"inputTokens": 10, "outputTokens": 5},
        }

        messages = [
            SystemMessage(content="You are a helpful assistant.", type="SystemMessage"),
            UserMessage(content="What's the weather in San Francisco?", source="user", type="UserMessage"),
        ]

        tool_schema = ToolSchema(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        )

        result = await client.create(messages, tools=[tool_schema])

        assert isinstance(result, CreateResult)
        assert isinstance(result.content, list)
        assert len(result.content) == 1
        assert isinstance(result.content[0], FunctionCall)
        assert result.content[0].name == "get_weather"
        assert result.finish_reason == "function_calls"

    @pytest.mark.asyncio
    async def test_create_stream(self, client):
        """Test create_stream method."""
        messages = [
            SystemMessage(content="You are a helpful assistant.", type="SystemMessage"),
            UserMessage(content="Hello, how are you?", source="user", type="UserMessage"),
        ]

        chunks = []
        async for chunk in client.create_stream(messages):
            chunks.append(chunk)

        # Check that we got the expected chunks
        assert len(chunks) == 5
        assert chunks[0] == "This "
        assert chunks[1] == "is "
        assert chunks[2] == "a test "
        assert chunks[3] == "response"
        assert isinstance(chunks[4], CreateResult)
        assert chunks[4].content == "This is a test response"
        assert chunks[4].finish_reason == "stop"
        assert chunks[4].usage.prompt_tokens == 10
        assert chunks[4].usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_create_stream_with_tool_calls(self, client):
        """Test create_stream method with tool calls."""
        # Mock the converse_stream response to include a tool call
        client.bedrock_runtime.converse_stream.return_value = {
            "stream": [
                {"contentBlockStart": {"start": {"toolUse": {"toolUseId": "call_123", "name": "get_weather"}}}},
                {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"location":"San'}}}},
                {"contentBlockDelta": {"delta": {"toolUse": {"input": ' Francisco","unit":"celsius"}'}}}},
                {"contentBlockStop": {}},
                {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
            ]
        }

        messages = [
            SystemMessage(content="You are a helpful assistant.", type="SystemMessage"),
            UserMessage(content="What's the weather in San Francisco?", source="user", type="UserMessage"),
        ]

        tool_schema = ToolSchema(
            name="get_weather",
            description="Get the current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        )

        chunks = []
        async for chunk in client.create_stream(messages, tools=[tool_schema]):
            chunks.append(chunk)

        # Check that we got the expected result
        assert len(chunks) == 1
        assert isinstance(chunks[0], CreateResult)
        assert isinstance(chunks[0].content, list)
        assert len(chunks[0].content) == 1
        assert isinstance(chunks[0].content[0], FunctionCall)
        assert chunks[0].content[0].id == "call_123"
        assert chunks[0].content[0].name == "get_weather"
        assert chunks[0].content[0].arguments == '{"location":"San Francisco","unit":"celsius"}'
        assert chunks[0].finish_reason == "function_calls"

    @pytest.mark.asyncio
    async def test_make_bedrock_api_call(self, client):
        """Test _make_bedrock_api_call method."""
        # Create a mock function that returns a value
        mock_fn = Mock(return_value={"result": "success"})

        # Call the method
        result = await client._make_bedrock_api_call(mock_fn)

        # Check that the function was called and the result is correct
        mock_fn.assert_called_once()
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_make_bedrock_api_call_with_cancellation(self, client):
        """Test _make_bedrock_api_call method with cancellation token."""
        # Create a mock function that returns a value
        mock_fn = Mock(return_value={"result": "success"})

        # Create a cancellation token
        cancellation_token = CancellationToken()

        # Call the method
        result = await client._make_bedrock_api_call(mock_fn, cancellation_token)

        # Check that the function was called and the result is correct
        mock_fn.assert_called_once()
        assert result == {"result": "success"}

    def test_model_info(self, client):
        """Test model_info property."""
        model_info = client.model_info

        assert model_info["vision"] is True
        assert model_info["function_calling"] is True
        assert model_info["json_output"] is True
        assert model_info["family"] == "Bedrock"

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test close method."""
        # This method doesn't do anything, but we should test it for coverage
        await client.close()
        # No assertions needed, just make sure it doesn't raise an exception
