# import datetime
import json

# import logging
# # import os
# # from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum

# # from typing import (
# #     Any,
# #     List,
# #     Optional,
# #     TypedDict,
# #     cast,
# # )
# import google.auth
import requests

# import vertexai
# import wikipedia
# from google.cloud import logging as google_cloud_logging
# from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
# from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI

# from langgraph.graph import END, StateGraph
# from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field, HttpUrl

# from pydantic_settings import BaseSettings, SettingsConfigDict
# from traceloop.sdk import Instruments, Traceloop
# from vertexai.preview import reasoning_engines
from app.landmark_description_generator.agent import (
    WikiTourState,
    autonomous_agent,
    initialize_agent,
)
from app.landmark_description_generator.utils import (
    get_body_speech_to_text,
    get_decoded_body_from_respopnse,
    get_headers_speech_to_text,
)

# from app.utils.gcs import create_bucket_if_not_exists
# from app.utils.tracing import CloudTraceLoggingSpanExporter
# from app.utils.typing import Feedback, InputChat, dumpd, dumps, ensure_valid_config


class PlaceTypes(StrEnum):
    art_gallery = "art_gallery"
    art_studio = "art_studio"
    cultural_landmark = "cultural_landmark"
    historical_place = "historical_place"
    monument = "monument"
    museum = "museum"
    sculpture = "sculpture"
    amusement_park = "amusement_park"
    botanical_garden = "botanical_garden"
    hiking_area = "hiking_area"
    historical_landmark = "historical_landmark"
    national_park = "national_park"
    park = "park"
    tourist_attraction = "tourist_attraction"
    bakery = "bakery"
    bar = "bar"
    bar_and_grill = "bar_and_grill"
    cafe = "cafe"
    cafeteria = "cafeteria"
    fast_food_restaurant = "fast_food_restaurant"
    fine_dining_restaurant = "fine_dining_restaurant"
    pizza_restaurant = "pizza_restaurant"
    pub = "pub"
    restaurant = "restaurant"


class Location(BaseModel):
    latitude: float
    longitude: float


class DisplayName(BaseModel):
    text: str
    language_code: str = Field(alias="languageCode")


class LocalizedText(BaseModel):
    text: str
    language_code: str = Field(alias="languageCode")


class Review(BaseModel):
    name: str
    text: LocalizedText
    original_text: LocalizedText = Field(alias="originalText")
    rating: float


class Reference(BaseModel):
    reviews: list[Review]
    places: str


class GenerativeSummary(BaseModel):
    overview: LocalizedText
    description: LocalizedText
    references: list[Reference]


class Place(BaseModel):
    id: str
    types: list[str]
    international_phone_number: str | None = Field(
        None, alias="internationalPhoneNumber"
    )
    formatted_address: str = Field(alias="formattedAddress")
    location: Location
    rating: float | None = Field(None, ge=0, le=5)
    website_uri: HttpUrl | None = Field(None, alias="websiteUri")
    user_rating_count: int | None = Field(None, alias="userRatingCount")
    display_name: DisplayName = Field(alias="displayName")
    reviews: list[Review] | None = None
    generative_summary: str | None = Field(None, alias="generativeSummary")

    def get_nearby_locations(
        self, radius: float, location_types: PlaceTypes
    ) -> "PlacesList":
        url = "https://places.googleapis.com/v1/places:searchNearby"

        field_mask = (
            "places.id,"
            "places.displayName,"
            "places.types,"
            "places.location,"
            "places.formattedAddress,"
            "places.priceLevel,"
            "places.rating,"
            "places.userRatingCount,"
            "places.websiteUri,"
            "places.internationalPhoneNumber"
            "places.reviews,"
            "places.generativeSummary"
        )

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": "AIzaSyDjoRzcHj72yIdFPSLTr4bJ5ywR7ltwVXY",  # TODO
            "X-Goog-FieldMask": field_mask,
        }

        data = {
            "locationRestriction": {
                "circle": {
                    "center": {
                        "latitude": self.location.latitude,
                        "longitude": self.location.longitude,
                    },
                    "radius": radius,
                }
            },
            "includedTypes": location_types,
            "maxResultCount": 20,
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return PlacesList.model_validate(response.json())

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error while searching for nearby locations: {e}") from e


class PlacesList(BaseModel):
    places: list[Place]

    def get_by_display_name(self, display_name: str) -> Place:
        try:
            return next(
                place
                for place in self.places
                if place.display_name.text == display_name
            )
        except StopIteration as e:
            raise IndexError(
                f"Place with display name '{display_name}' not found"
            ) from e
        except Exception as e:
            raise Exception(
                f"Error while searching for place with display name '{display_name}'"
            ) from e

    @classmethod
    def search_places(cls, query: str) -> "PlacesList":
        url = "https://places.googleapis.com/v1/places:searchText"

        field_mask = (
            "places.id,"
            "places.displayName,"
            "places.types,"
            "places.location,"
            "places.formattedAddress,"
            "places.priceLevel,"
            "places.rating,"
            "places.userRatingCount,"
            "places.websiteUri,"
            "places.internationalPhoneNumber,"
            "places.reviews,"
            "places.generativeSummary"
        )

        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": "AIzaSyDjoRzcHj72yIdFPSLTr4bJ5ywR7ltwVXY",  # TODO
            "X-Goog-FieldMask": field_mask,
        }

        data = {"textQuery": query}

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            array_places: PlacesList = PlacesList(places=[])

            if cls.model_validate(response.json()):
                array_places = cls.model_validate(response.json())

            return array_places
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error while searching for places: {e}") from e


class PlaceSimplified(BaseModel):
    name: str
    address: str
    rating: float | None = None
    website: HttpUrl | None = None
    phone_number: str | None = None
    types: list[str] | None = None

    @classmethod
    def from_place(cls, place: Place) -> "PlaceSimplified":
        return cls(
            name=place.display_name.text,
            address=place.formatted_address,
            rating=place.rating,
            website=place.website_uri,
            phone_number=place.international_phone_number,
        )


class PlacesListSimplified(BaseModel):
    places: list[PlaceSimplified]

    @classmethod
    def from_places_list(cls, places_list: PlacesList) -> "PlacesListSimplified":
        return cls(
            places=[PlaceSimplified.from_place(place) for place in places_list.places]
        )


def get_simple_description(place: Place) -> str:
    """
    Generate a simple description with the language model

    Args:
        topic: The topic to research

    Returns:
        A simple description of the topic
    """
    LOCATION = "us-central1"
    LLM = "gemini-2.0-flash-001"
    PROJECT = "qwiklabs-gcp-02-44d130f8f4a0"
    llm = ChatVertexAI(
        model=LLM, location=LOCATION, temperature=0, max_tokens=1024, project=PROJECT
    )
    messages = [
        (
            "system",
            "You have to create a simple description of the place provided by the user",
        ),
        ("human", place.model_dump_json()),
    ]
    description = llm.invoke(messages).content
    return description


def get_tour_guide_summary(topic: str) -> WikiTourState:
    """
    Run the agent to research a topic and return a custom state with all relevant information.

    Args:
        topic: The topic to research

    Returns:
        WikiTourState containing Wikipedia content and tour guide summary
    """
    initial_state = initialize_agent(topic)
    final_state = autonomous_agent.invoke(initial_state)

    # Return the complete custom state
    return final_state


def get_landmark_description(places: list[Place]):
    possible_types = [
        "museum",
        "historical_landmark",
        "historical_place",
        "monument",
        "sculpture",
        "cultural_landmark",
        "national_park",
        "point_of_interest"
    ]
    results = []

    for place in places:
        place_data = {
            "name": place.display_name.text,
            "address": place.formatted_address,
        }

        if bool(set(place.model_dump()["types"]) & set(possible_types)):
            # Use simple description for places with recognized types
            description = get_simple_description(place)
            place_data["description"] = description
        else:
            # Use tour guide summary for other places
            result = get_tour_guide_summary(place.display_name.text)

            if result["summary"]:
                place_data["description"] = result["summary"]
            else:
                place_data["description"] = "No description available"

        # Generate audio for the description (for all places)
        if (
            place_data.get("description")
            and place_data["description"] != "No description available"
        ):
            try:
                response = requests.post(
                    "https://texttospeech.googleapis.com/v1/text:synthesize?key=AIzaSyDjoRzcHj72yIdFPSLTr4bJ5ywR7ltwVXY",
                    json=get_body_speech_to_text(result["summary"]),
                    headers=get_headers_speech_to_text(),
                )
                print("RESPONSE")
                # Convert audio to base64 for embedding in the response
                audio_data = get_decoded_body_from_respopnse(response.json())
                audio_base64 = f"data:audio/mp3;base64,{audio_data.decode('utf-8')}"
                place_data["audio_file"] = audio_base64
            except Exception as e:
                print(f"Error generating audio for {place_data['name']}: {str(e)}")

        results.append(place_data)

    return {"places": results}


places = PlacesList.search_places("Bosco Verticale Milano")
print(get_landmark_description(places.places))
