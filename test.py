# %%
from pathlib import Path
import json
from pprint import pprint

import vertexai
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from vertexai import agent_engines


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
agents = [agent for agent in agent_engines.list()]

agent = agents[0]
print(agent)
