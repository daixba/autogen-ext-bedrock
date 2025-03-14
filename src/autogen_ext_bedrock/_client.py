import asyncio
import json
import logging
import os
import warnings
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Sequence, Union, cast

import boto3
from autogen_core import (
    EVENT_LOGGER_NAME,
    TRACE_LOGGER_NAME,
    CancellationToken,
    Component,
    FunctionCall,
    Image,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    CreateResult,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelCapabilities,  # type: ignore
    ModelInfo,
    RequestUsage,
    SystemMessage,
    UserMessage,
)
from autogen_core.tools import Tool, ToolSchema
from botocore.config import Config

from ._config import BedrockClientConfigurationConfigModel

event_logger = logging.getLogger(EVENT_LOGGER_NAME)
trace_logger = logging.getLogger(TRACE_LOGGER_NAME)


class BedrockChatCompletionClient(ChatCompletionClient):
    component_type = "model"
    component_config_schema = BedrockClientConfigurationConfigModel

    def __init__(self, **kwargs):
        """Chat completion client for Amazon Bedrock models.

        Args:
            model (str): The model id to use (e.g., "anthropic.claude-3-5-sonnet-20241022-v2:0")
            access_key (str, optional): AWS Access Key ID. Read from environment variable AWS_ACCESS_KEY_ID if not provided.
            secret_key (str, optional): AWS Secret Access Key. Read from environment variable AWS_SECRET_ACCESS_KEY if not provided.
            region (str, optional): AWS Region (e.g. "us-east-1"). Read from environment variable AWS_REGION if not provided.
            max_tokens (int, optional): Maximum tokens in the response. Default is 4096.
            temperature (float, optional): Controls randomness. Lower is more deterministic. Default is 1.0.
            top_p (float, optional): Controls diversity via nucleus sampling. Default is 1.0.

        Example:

        .. code-block:: python

            import asyncio
            from autogen_ext_bedrock import BedrockChatCompletionClient
            from autogen_core.models import UserMessage


            async def main():
                model_client = BedrockChatCompletionClient(
                    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                )

                result = await model_client.create(
                    [UserMessage(content="What is the capital of France?", source="user")]
                )  # type: ignore
                print(result)


            if __name__ == "__main__":
                asyncio.run(main())

        """
        config = BedrockClientConfigurationConfigModel.model_validate(kwargs)

        # # Set up AWS credentials and region
        self._access_key = config.access_key or os.environ.get("AWS_ACCESS_KEY_ID")
        self._secret_key = config.secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.region = config.region or os.environ.get("AWS_REGION", "us-east-1")

        timeout = config.timeout or 60
        max_retries = config.max_retries or 1
        boto_config = Config(connect_timeout=timeout, retries={"max_attempts": max_retries})

        # # Initialize Bedrock client
        self.bedrock_client = boto3.client(
            "bedrock",
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            config=boto_config,
            region_name=self.region,
        )

        self.bedrock_runtime = boto3.client(
            "bedrock-runtime",
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            config=boto_config,
            region_name=self.region,
        )

        # Set up model parameters
        self.model_id = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens
        self.top_p = config.top_p
        self.stop = config.stop
        model_list = self._list_models()
        modalities = model_list.get(self.model_id)["modalities"]
        vision = True if "IMAGE" in modalities else False
        self._model_info = ModelInfo(
            vision=vision,
            function_calling=True,
            json_output=True,
            family="Bedrock",
        )

        self._total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self._actual_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    def _list_models(self):
        model_list = {}
        try:
            profile_list = []
            if True:
                # List system defined inference profile IDs
                response = self.bedrock_client.list_inference_profiles(maxResults=1000, typeEquals="SYSTEM_DEFINED")

                profile_list = [p["inferenceProfileId"] for p in response["inferenceProfileSummaries"]]

            # List foundation models, only cares about text outputs here.
            response = self.bedrock_client.list_foundation_models(byOutputModality="TEXT")

            for model in response["modelSummaries"]:
                model_id = model.get("modelId", "N/A")
                stream_supported = model.get("responseStreamingSupported", True)
                status = model["modelLifecycle"].get("status", "ACTIVE")

                # currently, use this to filter out rerank models and legacy models
                if not stream_supported or status not in ["ACTIVE", "LEGACY"]:
                    continue

                inference_types = model.get("inferenceTypesSupported", [])
                input_modalities = model["inputModalities"]
                # Add on-demand model list
                if "ON_DEMAND" in inference_types:
                    model_list[model_id] = {"modalities": input_modalities}

                # Add cross-region inference model list.
                cr_inference_prefix = "apac" if self.region.startswith("ap-") else self.region[:2]
                profile_id = cr_inference_prefix + "." + model_id
                if profile_id in profile_list:
                    model_list[profile_id] = {"modalities": input_modalities}

        except Exception as e:
            event_logger.error(f"Unable to list models: {str(e)}")
        return model_list

    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> ChatCompletionClient:
        # return OpenAIChatCompletionClient(**config)
        raise NotImplementedError("BedrockChatCompletionClient does not support create_from_config")

    async def create(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
    ) -> CreateResult:
        """Create a chat completion using Bedrock's converse API."""
        # Convert messages to Bedrock format
        system_prompt, bedrock_messages = self._convert_messages_to_bedrock_format(messages)

        # Prepare inference config
        inference_config = {
            "maxTokens": self.max_tokens if self.max_tokens is not None else 4096,
            "temperature": self.temperature if self.temperature is not None else 0.7,
            "topP": self.top_p if self.top_p is not None else 0.9,
        }

        # Add stop sequences if provided
        if self.stop:
            inference_config["stopSequences"] = self.stop if isinstance(self.stop, list) else [self.stop]

        # Prepare tool configuration if tools are provided
        tool_config = None
        if tools:
            tool_config = self._prepare_tool_config(tools, json_output)

        try:
            # Make the API call
            request_args = {
                "modelId": self.model_id,
                "messages": bedrock_messages,
                "inferenceConfig": inference_config,
            }
            if system_prompt:
                request_args["system"] = [{"text": system_prompt}]

            # Add tool config if available
            if tool_config:
                request_args["toolConfig"] = tool_config

            # Add any extra arguments
            request_args.update(extra_create_args)

            # Make the API call
            response = await self._make_bedrock_api_call(
                lambda: self.bedrock_runtime.converse(**request_args), cancellation_token
            )

            # Process the response
            result = self._process_bedrock_response(response, messages, tools, json_output)

            return result

        except Exception:
            raise

    async def create_stream(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Sequence[Tool | ToolSchema] = [],
        json_output: Optional[bool] = None,
        # extra_create_args: Mapping[str, Any] = {},
        cancellation_token: Optional[CancellationToken] = None,
        max_consecutive_empty_chunk_tolerance: int = 0,
    ) -> AsyncGenerator[Union[str, CreateResult], None]:
        """Create a streaming chat completion using Bedrock's converse_stream API."""

        # Convert messages to Bedrock format
        system_prompt, bedrock_messages = self._convert_messages_to_bedrock_format(messages)

        # Prepare inference config
        inference_config = {
            "maxTokens": self.max_tokens if self.max_tokens is not None else 4096,
            "temperature": self.temperature if self.temperature is not None else 0.7,
            "topP": self.top_p if self.top_p is not None else 0.9,
        }

        # Add stop sequences if provided
        if self.stop:
            inference_config["stopSequences"] = self.stop if isinstance(self.stop, list) else [self.stop]

        # Prepare tool configuration if tools are provided
        tool_config = None
        if tools:
            tool_config = self._prepare_tool_config(tools, json_output)

        # Initialize variables for tracking the response
        full_text = ""
        usage = None

        try:
            # Make the API call
            request_args = {
                "modelId": self.model_id,
                "messages": bedrock_messages,
                "inferenceConfig": inference_config,
            }
            if system_prompt:
                request_args["system"] = [{"text": system_prompt}]

            # Add tool config if available
            if tool_config:
                request_args["toolConfig"] = tool_config

            # Make the API call
            streaming_response = await self._make_bedrock_api_call(
                lambda: self.bedrock_runtime.converse_stream(**request_args), cancellation_token
            )

            tool_calls = []
            tool_call = None
            # Process the streaming response
            for chunk in streaming_response["stream"]:
                # Check for cancellation
                # if cancellation_token:
                #     break

                if "contentBlockStart" in chunk:
                    # tool call start
                    delta = chunk["contentBlockStart"]["start"]
                    if "toolUse" in delta:
                        tool_call = FunctionCall(
                            id=delta["toolUse"]["toolUseId"],
                            name=delta["toolUse"]["name"],
                            arguments="",
                        )

                elif "contentBlockDelta" in chunk:
                    delta = chunk["contentBlockDelta"]["delta"]
                    # text = chunk["contentBlockDelta"]["delta"].get("text", "")
                    if "text" in delta:
                        full_text += delta["text"]
                        yield delta["text"]
                    elif "toolUse" in delta:
                        tool_call.arguments += delta["toolUse"]["input"]

                elif "contentBlockStop" in chunk:
                    if tool_call is not None:
                        tool_calls.append(tool_call)
                        tool_call = None

                # Process completion metadata
                elif "metadata" in chunk:
                    # Extract usage information if available
                    if "usage" in chunk["metadata"]:
                        usage = self._extract_usage_from_metadata(chunk["metadata"]["usage"])
                else:
                    # Handle other types of chunks as needed
                    pass

            if tool_calls:
                content = tool_calls
                finish_reason = "function_calls"
            else:
                content = full_text
                finish_reason = "stop"

            # Create the result.
            result = CreateResult(
                finish_reason=finish_reason,
                content=content,
                usage=usage,
                cached=False,
            )

            yield result

        except Exception:
            raise

    def actual_usage(self) -> RequestUsage:
        return self._actual_usage

    def total_usage(self) -> RequestUsage:
        return self._total_usage

    def count_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        return 0

    def remaining_tokens(self, messages: Sequence[LLMMessage], *, tools: Sequence[Tool | ToolSchema] = []) -> int:
        # token_limit = _model_info.get_token_limit(self._create_args["model"])
        # return token_limit - self.count_tokens(messages, tools=tools)
        return 0

    @property
    def capabilities(self) -> ModelCapabilities:  # type: ignore
        warnings.warn("capabilities is deprecated, use model_info instead", DeprecationWarning, stacklevel=2)
        return self._model_info

    @property
    def model_info(self) -> ModelInfo:
        return self._model_info

    async def close(self) -> None:
        # Do nothing
        pass

    # def _to_config(self) -> BedrockClientConfigurationConfigModel:
    #     copied_config = self._raw_config.copy()
    #     return BedrockClientConfigurationConfigModel(**copied_config)

    # @classmethod
    # def _from_config(cls, config: BedrockClientConfigurationConfigModel) -> Self:
    #     copied_config = config.model_copy().model_dump(exclude_none=True)
    #     return cls(**copied_config)

    def _to_config(self):
        raise NotImplementedError("BedrockClient does not support _to_config method.")

    @classmethod
    def _from_config(cls, config):
        # copied_config = config.model_copy().model_dump(exclude_none=True)
        # return cls(**copied_config)
        raise NotImplementedError("BedrockClient does not support _from_config method.")

    def _convert_messages_to_bedrock_format(self, messages: Sequence[LLMMessage]) -> tuple[str, List[Dict[str, Any]]]:
        """Convert AutoGen messages to Bedrock format.

        Return both system prompt and messages
        """
        bedrock_messages = []
        system_messages = []

        for message in messages:
            if isinstance(message, SystemMessage):
                system_messages.append(message.content)
                continue
            elif isinstance(message, UserMessage):
                content_blocks = []

                # Handle text content
                if message.content:
                    content_blocks.append({"text": message.content})

                # TODO: Handle images if present and model supports vision
                bedrock_message = {"role": "user", "content": content_blocks}
            elif isinstance(message, AssistantMessage):
                if isinstance(message.content, str) and message.content:
                    bedrock_message = {"role": "assistant", "content": [{"text": message.content}]}

                elif isinstance(message.content, list):
                    # Handle function calls if present
                    tool_calls = []
                    for tool_call in message.content:
                        assert isinstance(tool_call, FunctionCall), "Expect content as FunctionCall"
                        tool_input = json.loads(tool_call.arguments)
                        tool_calls.append(
                            {
                                "toolUse": {
                                    "toolUseId": tool_call.id,
                                    "name": tool_call.name,
                                    "input": tool_input,
                                }
                            }
                        )
                    bedrock_message = {"role": "assistant", "content": tool_calls}

            elif isinstance(message, FunctionExecutionResultMessage):
                for result in message.content:
                    bedrock_message = {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": result.call_id,
                                    "content": [{"text": result.content}],
                                }
                            }
                        ],
                    }
            else:
                # Default fallback
                bedrock_message = {"role": "user", "content": [{"text": str(message)}]}

            bedrock_messages.append(bedrock_message)

        system_prompt = "\n".join(system_messages) if system_messages else None
        return system_prompt, bedrock_messages

    def _prepare_tool_config(
        self, tools: Sequence[Tool | ToolSchema], json_output: Optional[bool] = None
    ) -> Dict[str, Any]:
        """Prepare tool configuration for Bedrock."""
        tool_config = {"tools": []}

        for tool in tools:
            if isinstance(tool, Tool):
                tool = tool.schema
            tool_spec = {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": {
                    "json": tool["parameters"],
                },
            }

            tool_config["tools"].append({"toolSpec": tool_spec})

        return tool_config

    async def _make_bedrock_api_call(self, api_call_fn, cancellation_token: Optional[CancellationToken] = None):
        """Make an API call to Bedrock."""
        try:
            # Create a future for the API call
            loop = asyncio.get_event_loop()
            future = asyncio.ensure_future(loop.run_in_executor(None, api_call_fn))

            # Link the future with the cancellation token if provided
            if cancellation_token is not None:
                cancellation_token.link_future(future)

            # Wait for the future to complete
            return await future
        except Exception as e:
            # If there's an exception, raise it directly
            raise e

    def _process_bedrock_response(
        self,
        response: Dict[str, Any],
        messages: Sequence[LLMMessage],
        tools: Sequence[Tool | ToolSchema],
        json_output: Optional[bool],
    ) -> CreateResult:
        """Process the response from Bedrock."""
        # Extract the message content
        output = response.get("output", {})

        message_data = output.get("message", {})
        content_blocks = message_data.get("content", [])
        tool_calls = []

        # Extract text content
        text_content = ""
        for block in content_blocks:
            if "text" in block:
                text_content += block["text"]
            elif "toolUse" in block:
                tool_block = block["toolUse"]
                tool_name = tool_block.get("name", "")
                tool_input = tool_block.get("input", {})
                # Convert to function call format
                function_call = FunctionCall(
                    id=tool_block.get("toolUseId", ""), name=tool_name, arguments=json.dumps(tool_input)
                )
                tool_calls.append(function_call)

        # Extract usage information
        usage_data = response.get("usage", {})
        usage = self._extract_usage_from_metadata(usage_data)

        # Store usage for tracking
        self._actual_usage = usage
        if not hasattr(self, "_total_usage"):
            self._total_usage = usage
        else:
            self._total_usage = RequestUsage(
                prompt_tokens=self._total_usage.prompt_tokens + usage.prompt_tokens,
                completion_tokens=self._total_usage.completion_tokens + usage.completion_tokens,
            )

        # self._total_usage = _add_usage(self._total_usage, usage)
        # self._actual_usage = _add_usage(self._actual_usage, usage)
        if tool_calls:
            content = tool_calls
            finish_reason = "function_calls"
        else:
            content = text_content
            finish_reason = "stop"

        # Create and return the result
        return CreateResult(
            finish_reason=finish_reason,
            content=content,
            usage=usage,
            cached=False,
            logprobs=None,
        )

    def _extract_usage_from_metadata(self, usage_data: Dict[str, Any]) -> RequestUsage:
        """Extract usage information from metadata."""
        prompt_tokens = usage_data.get("inputTokens", 0)
        completion_tokens = usage_data.get("outputTokens", 0)

        return RequestUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
