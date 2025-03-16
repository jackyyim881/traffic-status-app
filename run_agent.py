from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dotenv import load_dotenv
import os

load_dotenv()

XAI_API_KEY = os.getenv("X_API_KEY")

model = OpenAIModel(
    'grok-2-latest',
    provider=OpenAIProvider(base_url='https://api.x.ai/v1',
                            api_key=XAI_API_KEY),
)
agent = Agent(model)


result_sync = agent.run_sync('How is the weather in Hong Kong?')
print(result_sync.data)
# > Rome
