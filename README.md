# AutoGen Extension for Amazon Bedrock (experimental)

An quick experimental implementation of AutoGen Model extension for Amazon Bedrock models. Feel free to clone it and customize yourself.

## Installation

```bash
pip install git+https://github.com/daixba/autogen-ext-bedrock.git
```

## Quick Start

Set Up environment variables as below:

```bash
export AWS_ACCESS_KEY_ID=<your access key>
export AWS_SECRET_ACCESS_KEY=<your secret key>
export AWS_REGION=us-east-1
```

```python
# pip install -U "autogen-agentchat"
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext_bedrock import BedrockChatCompletionClient

model_client = BedrockChatCompletionClient(
    model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
)


async def main() -> None:
    agent = AssistantAgent("assistant", model_client=model_client)
    print(await agent.run(task="Say 'Hello World!'"))


asyncio.run(main())
```


## Known Issues

- Vision is not yet supported
- Token usage is not validated
- This model client doesn't work well in `Swarm` mode due to the mess of conversation history
