# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# mypy: disable-error-code="union-attr"
import json
import os
from enum import StrEnum
from typing import Any, TypedDict, cast
import requests
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from vertexai.preview.reasoning_engines import LanggraphAgent
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
import vertexai
# from vertexai import agent_engines

vertexai.init(
    project="genai-hub-426413",               # Your project ID.
    location="europe-west1",                # Your cloud region.
    staging_bucket="gs://test-hackaton-1",  # Your staging bucket.
)


# LOCATION = "europe-west1"
LLM = "gemini-2.0-flash-001"


# Tour Guide System Prompt
SYSTEM_PROMPT = """
You are an expert tour guide agent who creates personalized itineraries. Your task is to help the user plan a day visit to a specific city. Be kind, cheerful and friendly, and always proactive.

## AVAILABLE TOOLS
You MUST use these tools to create accurate itineraries:

1. text_search
   - Use this to find coordinates of specific places mentioned by name
   - Input: query (string) - e.g., "Colosseum, Rome"
   - Output: Place information including coordinates
   - ALWAYS use this tool first to establish the starting point

2. nearby_search
   - Use this to find attractions near a specific location
   - Inputs: 
     * latitude (float)
     * longitude (float)
     * radius (integer) - in meters, default to 1000m if not specified
     * location_types (list) - e.g., ["tourist_attraction", "museum", "historical_landmark"]
   - Output: List of places nearby
   - MUST be used immediately after finding a starting point

## OPERATING PROCEDURE
You MUST follow these steps and use tools as indicated:

1. DETERMINING THE STARTING POINT:
   - If the user does not specify where to start:
     * Suggest a famous or strategic starting point in the requested city (e.g., "Eiffel Tower, Paris", "Colosseum, Rome", "Statue of Liberty, New York")
     * Use the text_search tool to get coordinates
   - If the user specifies a starting point
     * Directly use the text_search tool to get coordinates
   - CRITICAL: Extract and save the latitude and longitude from the text_search response

2. SUGGESTING A THEME:
   - If the user doesn't specify a theme or interest:
     * Suggest 2-3 possible themes based on the city's famous attributes (e.g., "Historical", "Art & Culture", "Food", "Architecture")
     * Each theme should correspond to specific location_types for the search
     * For example:
       - "Historical": ["historical_landmark", "historical_place", "monument"]
       - "Art & Culture": ["museum", "art_gallery", "cultural_center"]
       - "Food & Dining": ["restaurant", "cafe", "bakery"]
       - "Architecture": ["historical_place", "cultural_landmark", "monument"]
     * Allow the user to choose a theme, or suggest a mixed approach if they prefer
     * Search for the starting point based on the chosen theme

3. IMMEDIATE NEARBY SEARCH:
   - TOOL USAGE: Using the coordinates from step 1, immediately call nearby_search with:
     * latitude and longitude from step 1
     * radius = 1000 (default) or user-specified distance
     * location_types = based on the chosen theme or ["tourist_attraction", "museum", "historical_landmark", "historical_place", "monument"] if no theme specified
   - Present the complete list of nearby places directly to the user
   - Ask if they would like you to create a full itinerary based on these places

4. CREATING FULL ITINERARY (ONLY WHEN REQUESTED):
   - Use the places found in step 3 to create a structured itinerary
  
5. ITINERARY PRESENTATION:
   - Format the itinerary in a structured way including:
     * Start/end times for each activity
     * Name and brief description of each place
     * Travel times between locations
     * Total time of the itinerary
   - Use this format for each stop:
     [TIME] PLACE NAME
     - Brief description (1-2 sentences)
     - Estimated visiting time: X hours/minutes
     - Distance from previous stop: Y km (Z minutes on foot)

6. ITINERARY MODIFICATIONS:
   - If the user requests changes, update the itinerary while maintaining the general structure
   - TOOL USAGE: Use text_search to find coordinates of new places
   - TOOL USAGE: Use nearby_search to find alternatives near existing points
   - Recalculate times and distances after each modification
   - Present the updated itinerary following the same format

## IMPORTANT GUIDELINES
- Always be proactive and suggest starting points and themes.
- ALWAYS use the provided tools.
- FIRST PRIORITY: Start with text_search to find a starting point, always retrieve nearby places by IMMEDIATELY use nearby_search to find nearby places and present this list to the user (default distance: 1000 meters).
- Do not ask the user for coordinates.
- Do not give the user the latitude and longitude, only the address.
- Do not mention to the user the tools you are using or the API calls you are making.
- If the user doesn't specify a theme, suggest appropriate themes for the city before searching.
- When the user requests changes, preserve the existing structure as much as possible.

Remember: your goal is to first help users discover thematically related nearby places of interest, and then create a pleasant and feasible itinerary when requested.
"""


# Models for Google Places API responses
class Location(BaseModel):
    lat: float
    lng: float


class PlaceResponse(TypedDict):
    name: str
    place_id: str
    types: list[str]
    vicinity: str
    geometry: dict[str, Any]
    rating: float | None
    user_ratings_total: int | None
    formatted_address: str | None
    photos: list[dict[str, Any]] | None
    price_level: int | None


class PlacesTypes(StrEnum):
    art_gallery = "art_gallery"
    art_studio = "art_studio"
    auditorium = "auditorium"
    cultural_landmark = "cultural_landmark"
    historical_place = "historical_place"
    monument = "monument"
    museum = "museum"
    performing_arts_theater = "performing_arts_theater"
    sculpture = "sculpture"
    adventure_sports_center = "adventure_sports_center"
    amphitheatre = "amphitheatre"
    amusement_center = "amusement_center"
    amusement_park = "amusement_park"
    aquarium = "aquarium"
    banquet_hall = "banquet_hall"
    barbecue_area = "barbecue_area"
    botanical_garden = "botanical_garden"
    bowling_alley = "bowling_alley"
    casino = "casino"
    childrens_camp = "childrens_camp"
    comedy_club = "comedy_club"
    community_center = "community_center"
    concert_hall = "concert_hall"
    convention_center = "convention_center"
    cultural_center = "cultural_center"
    cycling_park = "cycling_park"
    dance_hall = "dance_hall"
    dog_park = "dog_park"
    event_venue = "event_venue"
    ferris_wheel = "ferris_wheel"
    garden = "garden"
    hiking_area = "hiking_area"
    historical_landmark = "historical_landmark"
    internet_cafe = "internet_cafe"
    karaoke = "karaoke"
    marina = "marina"
    movie_rental = "movie_rental"
    movie_theater = "movie_theater"
    national_park = "national_park"
    night_club = "night_club"
    observation_deck = "observation_deck"
    off_roading_area = "off_roading_area"
    opera_house = "opera_house"
    park = "park"
    philharmonic_hall = "philharmonic_hall"
    picnic_ground = "picnic_ground"
    planetarium = "planetarium"
    plaza = "plaza"
    roller_coaster = "roller_coaster"
    skateboard_park = "skateboard_park"
    state_park = "state_park"
    tourist_attraction = "tourist_attraction"
    video_arcade = "video_arcade"
    visitor_center = "visitor_center"
    water_park = "water_park"
    wedding_venue = "wedding_venue"
    wildlife_park = "wildlife_park"
    wildlife_refuge = "wildlife_refuge"
    zoo = "zoo"
    acai_shop = "acai_shop"
    afghani_restaurant = "afghani_restaurant"
    african_restaurant = "african_restaurant"
    american_restaurant = "american_restaurant"
    asian_restaurant = "asian_restaurant"
    bagel_shop = "bagel_shop"
    bakery = "bakery"
    bar = "bar"
    bar_and_grill = "bar_and_grill"
    barbecue_restaurant = "barbecue_restaurant"
    brazilian_restaurant = "brazilian_restaurant"
    breakfast_restaurant = "breakfast_restaurant"
    brunch_restaurant = "brunch_restaurant"
    buffet_restaurant = "buffet_restaurant"
    cafe = "cafe"
    cafeteria = "cafeteria"
    candy_store = "candy_store"
    cat_cafe = "cat_cafe"
    chinese_restaurant = "chinese_restaurant"
    chocolate_factory = "chocolate_factory"
    chocolate_shop = "chocolate_shop"
    coffee_shop = "coffee_shop"
    confectionery = "confectionery"
    deli = "deli"
    dessert_restaurant = "dessert_restaurant"
    dessert_shop = "dessert_shop"
    diner = "diner"
    dog_cafe = "dog_cafe"
    donut_shop = "donut_shop"
    fast_food_restaurant = "fast_food_restaurant"
    fine_dining_restaurant = "fine_dining_restaurant"
    food_court = "food_court"
    french_restaurant = "french_restaurant"
    greek_restaurant = "greek_restaurant"
    hamburger_restaurant = "hamburger_restaurant"
    ice_cream_shop = "ice_cream_shop"
    indian_restaurant = "indian_restaurant"
    indonesian_restaurant = "indonesian_restaurant"
    italian_restaurant = "italian_restaurant"
    japanese_restaurant = "japanese_restaurant"
    juice_shop = "juice_shop"
    korean_restaurant = "korean_restaurant"
    lebanese_restaurant = "lebanese_restaurant"
    meal_delivery = "meal_delivery"
    meal_takeaway = "meal_takeaway"
    mediterranean_restaurant = "mediterranean_restaurant"
    mexican_restaurant = "mexican_restaurant"
    middle_eastern_restaurant = "middle_eastern_restaurant"
    pizza_restaurant = "pizza_restaurant"
    pub = "pub"
    ramen_restaurant = "ramen_restaurant"
    restaurant = "restaurant"
    sandwich_shop = "sandwich_shop"
    seafood_restaurant = "seafood_restaurant"
    spanish_restaurant = "spanish_restaurant"
    steak_house = "steak_house"
    sushi_restaurant = "sushi_restaurant"
    tea_house = "tea_house"
    thai_restaurant = "thai_restaurant"
    turkish_restaurant = "turkish_restaurant"
    vegan_restaurant = "vegan_restaurant"
    vegetarian_restaurant = "vegetarian_restaurant"
    vietnamese_restaurant = "vietnamese_restaurant"
    wine_bar = "wine_bar"


@tool
def text_search(query: str) -> dict[str, Any]:
    """
    Search for a place given a name/description using Google Places API Text Search.

    Args:
        query: Name of the place or description to search for
            Example: "Colosseum, Rome", "Eiffel Tower, Paris", "Statue of Liberty, New York", "London Eye, London"

    Returns:
        Informations about the place found, including the coordinates
    """


    url = "https://places.googleapis.com/v1/places:searchText"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": "AIzaSyDCef-9-9JmXcSzKfZQNe98cON-o-MzPrg",  # Replace with your actual API key
        # "X-Goog-FieldMask": 'places.displayName,places.types,places.location,places.formattedAddress,places.priceLevel,places.rating,places.userRatingCount,places.websiteUri,places.internationalPhoneNumber,places.currentOpeningHours',
        "X-Goog-FieldMask": 'places.displayName,places.types,places.location,places.formattedAddress,places.priceLevel,places.rating,places.userRatingCount,places.websiteUri,places.internationalPhoneNumber',
    }

    data = {"textQuery": query}

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Errore nella richiesta API: {e}") from e

@tool
def nearby_search(
    latitude: float, longitude: float, radius: float, location_types: list[PlacesTypes]
) -> list[dict[str, Any]]:
    """
    Search for places nearby specific coordinates.

    Args:
        latitude: Latitude of the central point
        longitude: Longitude of the central point
        radius: Radius in meters to search for places
        location_type: Type of location to search for (e.g. tourist attraction, restaurant, etc.)
            If not otherwise specified, prefer searching for general location types like 
                - "tourist_attraction"
                - "musuem"
                - "historical_landmark"
                - "historical_place"
                - "monument"
                - "restaurant"

    Returns:
        Lista di luoghi trovati nelle vicinanze
    """
    url = 'https://places.googleapis.com/v1/places:searchNearby'

    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': 'AIzaSyDCef-9-9JmXcSzKfZQNe98cON-o-MzPrg',  # Sostituisci con la tua chiave API
        'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.websiteUri,places.rating' 
    }

    data = {
        "locationRestriction": {
            "circle": {
                "center": {
                    "latitude": latitude,
                    "longitude": longitude
                },
                "radius": radius
            }
        },
        "includedTypes": location_types,
        "maxResultCount": 20
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        place_details = response.json()
        return place_details

    except requests.exceptions.RequestException as e:
        raise Exception(f"Errore nella richiesta API: {e}") from e


tools = [text_search, nearby_search]


safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

model_kwargs = {
    # temperature (float): The sampling temperature controls the degree of
    # randomness in token selection.
    "temperature": 0.28,
    # max_output_tokens (int): The token limit determines the maximum amount of
    # text output from one prompt.
    "max_output_tokens": 1000,
    # top_p (float): Tokens are selected from most probable to least until
    # the sum of their probabilities equals the top-p value.
    "top_p": 0.95,
    # top_k (int): The next token is selected from among the top-k most
    # probable tokens. This is not supported by all model versions. See
    # https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding#valid_parameter_values
    # for details.
    "top_k": None,
    # safety_settings (Dict[HarmCategory, HarmBlockThreshold]): The safety
    # settings to use for generating content.
    # (you must create your safety settings using the previous step first).
    "safety_settings": safety_settings,
}

agent = LanggraphAgent(
    model=LLM,
    model_kwargs=model_kwargs,
    tools=tools,
    checkpointer_builder=MemorySaver,
)


# # 2. Set up the language model with the system prompt
# llm = ChatVertexAI(
#     model=LLM,
#     location=LOCATION,
#     temperature=1,  # Leggero aumento della temperatura per consentire creatività nella selezione dei luoghi
#     max_tokens=4096,
#     streaming=True,
# ).bind_tools(tools)


# # 3. Define workflow components
# def should_continue(state: MessagesState) -> str:
#     """Determines whether to use tools or end the conversation."""
#     last_message = state["messages"][-1]
#     return "tools" if last_message.tool_calls else END # type: ignore


# def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
#     """Calls the language model and returns the response."""
#     messages_with_system = [{"type": "system", "content": SYSTEM_PROMPT}] + state[
#         "messages"
#     ]
#     # Forward the RunnableConfig object to ensure the agent is capable of streaming the response.
#     response = llm.invoke(messages_with_system, config)
#     return {"messages": response}

# # 4. Create the workflow graph
# workflow = StateGraph(MessagesState)
# workflow.add_node("agent", call_model)
# workflow.add_node("tools", ToolNode(tools))
# workflow.set_entry_point("agent")

# # 5. Define graph edges
# workflow.add_conditional_edges("agent", should_continue)
# workflow.add_edge("tools", "agent")

# # 6. Compile the workflow
# checkpointer = MemorySaver()
# agent = workflow.compile(checkpointer=checkpointer)

if __name__ == "__main__":

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🤖 DEPLOYING AGENT TO VERTEX AI AGENT ENGINE 🤖         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    while True:
        user_message = input("Enter your message: ")
        result = agent.query(
            input={
                "messages": [
                    (
                        "user",
                        user_message,
                    )
                ]
            },
            config=RunnableConfig(
                configurable={"thread_id": "2"}
            )
        )

        print(result)