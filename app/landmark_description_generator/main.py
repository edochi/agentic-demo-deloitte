import datetime
import json
import logging
import os
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import (
    Any,
    TypedDict,
    cast,
)

import google.auth
import requests
import vertexai
from google.cloud import logging as google_cloud_logging
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict
from traceloop.sdk import Instruments, Traceloop
from vertexai.preview import reasoning_engines

from app.utils.gcs import create_bucket_if_not_exists
from app.utils.tracing import CloudTraceLoggingSpanExporter
from app.utils.typing import Feedback, InputChat, dumpd, dumps, ensure_valid_config


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
            return cls.model_validate(response.json())
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


# def text_search(query: str) -> PlacesList:
#     """
#     Search for a place given a name/description using Google Places API Text Search.

#     Args:
#         query: Name of the place or description to search for
#             Example: "Colosseum, Rome", "Eiffel Tower, Paris", "Statue of Liberty, New York", "London Eye, London"

#     Returns:
#         Informations about the place found, including the coordinates
#     """

#     url = "https://places.googleapis.com/v1/places:searchText"

#     field_mask = (
#         "places.id,"
#         "places.displayName,"
#         "places.types,"
#         "places.location,"
#         "places.formattedAddress,"
#         "places.priceLevel,"
#         "places.rating,"
#         "places.userRatingCount,"
#         "places.websiteUri,"
#         "places.internationalPhoneNumber"
#     )

#     headers = {
#         "Content-Type": "application/json",
#         "X-Goog-Api-Key": "AIzaSyDCef-9-9JmXcSzKfZQNe98cON-o-MzPrg",  # Replace with your actual API key
#         "X-Goog-FieldMask": field_mask,
#     }

#     data = {"textQuery": query}

#     try:
#         response = requests.post(url, headers=headers, data=json.dumps(data))
#         response.raise_for_status()  # Raise an exception for bad status codes

#         # Place.model_validate(response.json())
#         return PlacesList.model_validate(response.json())
#     except requests.exceptions.RequestException as e:
#         raise Exception(f"Errore nella richiesta API: {e}") from e


# def nearby_search(
#     place_name: str, radius: float, location_types: list[PlaceTypes]
# ) -> PlacesList:
#     """
#     Search for places nearby specific coordinates.

#     Args:
#         latitude: Latitude of the central point
#         longitude: Longitude of the central point
#         radius: Radius in meters to search for places
#         location_type: Type of location to search for (e.g. tourist attraction, restaurant, etc.)
#             If not otherwise specified, prefer searching for general location types like
#                 - "tourist_attraction"
#                 - "musuem"
#                 - "historical_landmark"
#                 - "historical_place"
#                 - "monument"
#                 - "restaurant"

#     Returns:
#         Lista di luoghi trovati nelle vicinanze
#     """
#     url = "https://places.googleapis.com/v1/places:searchNearby"

#     field_mask = (
#         "places.id,"
#         "places.displayName,"
#         "places.types,"
#         "places.location,"
#         "places.formattedAddress,"
#         "places.priceLevel,"
#         "places.rating,"
#         "places.userRatingCount,"
#         "places.websiteUri,"
#         "places.internationalPhoneNumber"
#     )

#     headers = {
#         "Content-Type": "application/json",
#         "X-Goog-Api-Key": "AIzaSyDCef-9-9JmXcSzKfZQNe98cON-o-MzPrg",  # Sostituisci con la tua chiave API
#         "X-Goog-FieldMask": field_mask,
#     }

#     data = {
#         "locationRestriction": {
#             "circle": {
#                 "center": {"latitude": latitude, "longitude": longitude},
#                 "radius": radius,
#             }
#         },
#         "includedTypes": location_types,
#         "maxResultCount": 20,
#     }

#     try:
#         response = requests.post(url, headers=headers, data=json.dumps(data))
#         response.raise_for_status()
#         return PlacesList.model_validate(response.json())

#     except requests.exceptions.RequestException as e:
#         raise Exception(f"Errore nella richiesta API: {e}") from e


#if __name__ == "__main__":
    # Cerca un luogo dato un nome/descrizione
    # place_info = text_search("Rome, Italy")
    # print(json.dumps(place_info, indent=2))

    # print(text_search("colosseo, roma"))

    # locations = nearby_search(
    #     41.890251, 12.492373, 1000, [PlaceTypes.tourist_attraction]
    # )

    # print(locations.get_by_display_name("Roman Forum"))

    # print(text_search("colosseo, roma"))

    #places = PlacesList.search_places("colosseo roma")

    #print(places.places[0].reviews)

    #simplified_places = PlacesListSimplified.from_places_list(places)

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from typing import TypedDict, List, Optional
import wikipedia
import requests
import base64
from app.landmark_description_generator.agent import WikiTourState, initialize_agent, autonomous_agent

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
    model=LLM, 
    location=LOCATION, 
    temperature=0, 
    max_tokens=1024, 
    project=PROJECT
    )
    messages = [
        ("system",
        "You have to create a simple description of the place provided by the user",
        ),
        ("human", place.model_dump_json())
        ]
    description=llm.invoke(messages).content
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
    possible_types= ["museum", "historical_landmark", "historical_place", "monument", "sculpture", "cultural_landmark", "national_park"]
    for place in places:
        if bool(set(place.model_dump()['types']) & set(possible_types)):
            result = get_tour_guide_summary(place.model_dump()['display_name'])
            json={
                "input": {
                    "text": result['summary']
                },
                "voice": {
                    "languageCode": "en-gb",
                    "name": "en-GB-Standard-A",
                    "ssmlGender": "FEMALE"
                },
                "audioConfig": {
                    "audioEncoding": "MP3"
                }
            }
            headers={"Authorization": "Bearer ya29.a0AeXRPp4cM8sBM--epXkZNml62LkSW5gSUc3mBINDrIof_lqq8Yb2kQ3aWOEM_43WarVZosTzY-gRNemJrKmpgcvkauVEtBJWJQ-rF8MszUkrTZuHJbqwW20IxLFxvmegMj8CTj1v57ice5GEqnF85tDR_FkWXRV6Xxg7HalusJbO5gLvG3zuJkNOORPBJT6NSzcx6HBK4fvkP-WeX18rtwWxKYtRKHyHG3obLAEh9fP5jLokOZqO3dH5YvuD9gPjSI0fVS_AEu3phXyzWtXjkSSlryT4Wz9q69JxqDyOqiO1yD2eO5yUsIBu1L0k2IR3LvgQ5o7Dbj6hPYQaBxk8VJOiChNH0iarFc2oEIge_uXLJSj8IyqlLYi4xn6PFJBO9QDG-d_UBRd_eTw9XxYIipSibNU51dtLgVfdNwgaCgYKASsSARISFQHGX2MiiToxMRCwu_R7Qv8bkzYCFw0430",
                    "x-goog-user-project": 'qwiklabs-gcp-02-44d130f8f4a0',
                    "Content-Type": "application/json; charset=utf-8"}
            data = requests.post('https://texttospeech.googleapis.com/v1/text:synthesize',json=json, headers=headers)
            decoded_data = base64.b64decode(data.json()['audioContent'])
            with open ('test_audio.mp3', 'wb') as f_audio:
                f_audio.write(decoded_data)
            return result['wiki_content']
        else:
            description = get_simple_description(place)
            return description

places = PlacesList.search_places("milano bosco verticale")
print(get_landmark_description(places.places))

