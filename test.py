# %%
from pathlib import Path
import json
from pprint import pprint

import vertexai
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from vertexai import agent_engines
from langchain_core.runnables import RunnableConfig


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    project_id: str = Field(alias="PROJECT_ID")
    location: str = Field(alias="LOCATION")
    staging_bucket: str = Field(alias="STAGING_BUCKET")


SETTINGS = Settings()  # type: ignore


vertexai.init(
    project=SETTINGS.project_id,
    location=SETTINGS.location,
    staging_bucket=f"gs://{SETTINGS.staging_bucket}",
)

# %%
# agents = [agent for agent in agent_engines.list()]

# agent = agents[0]

agent = agent_engines.get('projects/889354480933/locations/us-central1/reasoningEngines/1016133462018490368')


print(agent.query(
            input={
            "messages": [
                (
                    "user",
                    'what can you do?',
                )
            ]
        },
        config=RunnableConfig(configurable={"thread_id": "2"}),
        stream_mode="values",
))
