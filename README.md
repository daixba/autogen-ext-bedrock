# AutoGen Extension for Amazon Bedrock


## Installation

```bash
pip install autogen-ext-bedrock
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