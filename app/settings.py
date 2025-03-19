import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logging.basicConfig(
    level=logging.INFO,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    project_id: str = Field(alias="PROJECT_ID")
    location: str = Field(alias="LOCATION")
    staging_bucket: str = Field(alias="STAGING_BUCKET")
    places_api_key: str = Field(alias="PLACES_API_KEY")

    def get_env_vars_key_value_format(self):
        return ",".join(
            f"{key}={value}" for key, value in self.model_config.items()
        )


SETTINGS = Settings()  # type: ignore