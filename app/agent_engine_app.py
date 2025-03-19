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
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Annotated,
    Any,
    Literal,
    TypedDict,
    TypeVar,
    overload,
)

import vertexai
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.runnables import (
    RunnableConfig,
)
from langchain_core.tools import tool
from langchain_core.tools.base import InjectedToolCallId
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph.message import add_messages
from langgraph.managed import IsLastStep, RemainingSteps
from langgraph.prebuilt import InjectedState, create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.classes import PlacesList, PlacesListSimplified, PlaceTypes

# from langgraph.checkpoint.memory import MemorySaver

TOther = TypeVar("TOther")
TBaseModel = TypeVar("TBaseModel", bound=BaseModel)


class FieldGetter:
    """
    A utility class to retrieve and validate fields from a dictionary.

    Attributes:
        input_dict (Mapping[str, Any]): The input dictionary from which fields are
            retrieved.
    """

    def __init__(self, input_dict: Mapping[str, Any]):
        self.input_dict = input_dict

    @overload
    def get_field(
        self,
        field: str,
        output_type: Callable[[Any], TOther] = str,
        raise_error_if_missing: Literal[True] = True,
    ) -> TOther: ...

    @overload
    def get_field(
        self,
        field: str,
        output_type: Callable[[Any], TOther] = str,
        raise_error_if_missing: Literal[False] = False,
    ) -> TOther | None: ...

    @overload
    def get_field(
        self,
        field: str,
        output_type: type[TBaseModel],
        raise_error_if_missing: Literal[True] = True,
    ) -> TBaseModel: ...

    @overload
    def get_field(
        self,
        field: str,
        output_type: type[TBaseModel],
        raise_error_if_missing: Literal[False] = False,
    ) -> TBaseModel | None: ...

    def get_field(
        self,
        field: str,
        output_type: Callable[[Any], TOther] | type[TBaseModel] = str,
        raise_error_if_missing: bool = True,
    ) -> TOther | TBaseModel | None:
        """
        Retrieve a field from the input dictionary and convert it to the specified
            output type.

        Args:
            field (str): The key of the field to retrieve from the input dictionary.
            output_type (Callable[[Any], TOther] | Type[TBaseModel], optional):
                The type to which the field value should be converted. Defaults to str.
            raise_error_if_missing (bool, optional): Whether to raise a KeyError if the
                field is missing. Defaults to True.

        Returns:
            TOther | TBaseModel | None: The value of the field converted to the
                specified output type, or None if the field is missing and
                raise_error_if_missing is False.

        Raises:
            KeyError: If the field is missing and raise_error_if_missing is True.
        """
        value = self.input_dict.get(field)
        if value is None:
            if raise_error_if_missing:
                raise KeyError(
                    f"In node {inspect.stack()[1].function} "
                    f"the state field {field} is missing"
                )
            return None
        if isinstance(output_type, BaseModel):
            if isinstance(value, str):
                return output_type.model_validate_json(value)
            if isinstance(value, dict):
                return output_type.model_validate(value)
            if isinstance(value, BaseModel):
                return output_type.model_validate(value)
        return TypeAdapter(output_type).validate_python(value)


vertexai.init(
    project="genai-hub-426413",  # Your project ID.
    location="europe-west1",  # Your cloud region.
    staging_bucket="gs://test-hackaton-1",  # Your staging bucket.
)


# LOCATION = "europe-west1"
LLM = "gemini-2.0-flash-001"


# Tour Guide System Prompt
SYSTEM_PROMPT = """
You are an expert tour guide agent who creates personalized itineraries. Your task is to help the user plan a day visit to a specific city. Be kind, cheerful and friendly, and always proactive.

You must, given a starting point, determine a tour to guide the user, propose it to them, and make changes as requested.

Techincally, the tour is the list of the display names of the places to visit.

## OPERATING PROCEDURE
You MUST follow these steps and use tools as indicated:

1. SUGGESTING A THEME:
    If the user doesn't specify a theme or interest:
    - Suggest 2-3 possible themes based on the city's famous attributes (e.g., "Historical", "Art & Culture", "Food", "Architecture")
    - Each theme should correspond to specific "location_types" for the search. For example:
       - "Historical": places like "historical_landmark", "historical_place", "monument", cultural_landmark", ...
       - "Art & Culture": places like "museum", "art_gallery", "cultural_center", cultural_landmark", ...
       - "Food & Dining": places like "restaurant", "cafe", "bakery", ...
       - "Shopping": places like "clothing_store", "shoe_store", "jewelry_store", "shopping_mall", ...
    - Allow the user to choose a theme, or suggest a mixed approach if they prefer
    - Search for the starting point based on the chosen theme

2. DETERMINING THE STARTING POINT:
   - If the user does not specify where to start: suggest a famous or strategic starting point in the requested city (e.g., a famous landmark, if the user wants to visit something historical or cultural, a famous cafe if he wants to a have some food), ask if they agree, and if it is ok for them, use it as the starting point for the itinerary
   - If the user specifies a starting point, use it as the starting point

   In any case, remember to get the Display Name of the place using the appropriate tool before adding it to the tour list.

   YOU MUST ALWAYS USE THE "place_search" TOOL TO SEARCH FOR PLACES
   YOU MUST ALWAYS USE THE "add_place_to_tour" TOOL TO ADD PLACES TO THE TOUR

3. IMMEDIATE NEARBY SEARCH:
    - After finding the starting point, IMMEDIATELY search for nearby places of interest based on the theme chosen (if any).
    - Start searching within a radius of 1000 meters, but if nothing is found, or nothing seems interesting, ask the user if they want to expand the search radius or change the theme.
    - Present the list of places found to the user, and allow them to choose which places to add to the itinerary.
    - If the user asks you to proceed with the itinerary creation by yourself, recursively search for nearby places of interest around the places already found, and add them to the itinerary. Go on until the itinerary is composed of at least 6 places.

4. ITINERARY PRESENTATION:
   - When the itinerary is complete, present it to the user in a structured way, including the names of the places, their addresses, and the order in which they will be visited.
   - Provide a summary of the itinerary, including the total time and distance of the tour.
   - Ask the user if they are satisfied with the itinerary, and if they want to make any changes.

5. ITINERARY MODIFICATIONS:
   - If the user requests changes, update the itinerary while maintaining the general structure

## IMPORTANT GUIDELINES
- Always be proactive and suggest starting points and themes.
- ALWAYS use the provided tools.
- If the user doesn't specify a theme, suggest appropriate themes for the city before searching.
- When the user requests changes, preserve the existing structure as much as possible.
- Always communicate with the user using the language they are using.
- TO CALL A TOOL, USE THE TOOL SCHEMA AND THE TOOL NAME, do not use "print.(default_api...)"
"""


class AgentState(TypedDict, total=False):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    is_last_step: IsLastStep
    remaining_steps: RemainingSteps
    places_list: PlacesList
    tour: list[str]
    places_list_tour: PlacesList


@tool
def place_search(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
) -> Command:
    """
    Search for a place given a name/description using Google Places API Text Search.

    Args:
        query: Name of the place or description to search for
            Example: "Colosseum, Rome", "Eiffel Tower, Paris", "Statue of Liberty, New York", "London Eye, London"

    Returns:
        Informations about the place found, including the coordinates
    """

    # print(state)

    places_list = PlacesList.search_places(query)
    places_simplified = PlacesListSimplified.from_places_list(places_list)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=places_simplified.get_structured_string(),
                    tool_call_id=tool_call_id,
                )
            ],
            "places_list": places_list,
        }
    )


@tool
def places_nearby(
    place_display_name: str,
    location_types: list[PlaceTypes],
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
    radius: float = 1000.0,
) -> Command:
    """
    Search for places nearby a specific place.

    Args:
        place_display_name: Display Name of the place around which to search for other places
            Careful: This is the display name of the place, not the name of the place
        radius: Radius in meters to search for places
        location_type: Type of location to search for (e.g. tourist attraction, restaurant, etc.)

    Returns:
        List of places found nearby
    """

    places_list = FieldGetter(state).get_field(
        "places_list", PlacesList, raise_error_if_missing=False
    )

    if places_list is None:
        places_list = PlacesList()
    # else:
    #     places_list = PlacesList(places=[Place.model_validate(place_in_list) for place_in_list in places_list.places])

    try:
        place = places_list.get_by_display_name(place_display_name)
        print(f"Found place {place} - OK")
    except IndexError as e:
        try:
            places_result = PlacesList.search_places(place_display_name)
            if not places_result.places:
                raise ValueError(f"No places found for {place_display_name}")
            place = places_result.places[0]
            places_list.append(place)
            print(f"Found place {place} - OK 2")
            print(f"Places list: {places_list} - OK 2")
        except ValidationError as e:
            raise Exception(
                f"Error validating place data for {place_display_name}: {e}"
            ) from e
        except Exception as e:
            raise Exception(
                f"Error while searching for place with display name {place_display_name}: {e}"
            ) from e

    nearby_places = place.get_nearby_locations(radius, location_types)
    places_list.extend(nearby_places)
    nearby_places_simplified = PlacesListSimplified.from_places_list(nearby_places)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=nearby_places_simplified.get_structured_string(),
                    tool_call_id=tool_call_id,
                )
            ],
            "places_list": places_list,
        }
    )


@tool
def add_place_to_tour(
    place_display_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
) -> Command:
    """
    Add a place to the tour list.

    Args:
        place_display_name: Display Name of the place to add to the tour
            The Display Name, if not already known, can be retrieved using the place_search tool

    Returns:
        Updated tour list
    """

    tour = FieldGetter(state).get_field("tour", list[str], raise_error_if_missing=False)
    if tour is None:
        tour = []

    places_list = FieldGetter(state).get_field(
        "places_list", PlacesList, raise_error_if_missing=False
    )
    if places_list is None:
        places_list = PlacesList()

    try:
        place = places_list.get_by_display_name(place_display_name)

    except IndexError as e:
        try:
            places_result = PlacesList.search_places(place_display_name)
            if not places_result.places:
                raise ValueError(f"No places found for {place_display_name}")
            place = places_result.places[0]
            places_list.append(place)
        except ValidationError as e:
            raise Exception(
                f"Error validating place data for {place_display_name}: {e}"
            ) from e
        except Exception as e:
            raise Exception(
                f"Error while searching for place with display name {place_display_name}: {e}"
            ) from e

    tour.append(place.displayName.text)
    places_list_tour = places_list.get_by_display_names(tour)

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Added {place_display_name} to the tour list\n\nThis is the updated tour:\n{tour}",
                    tool_call_id=tool_call_id,
                )
            ],
            "tour": tour,
            "places_list": places_list,
            "places_list_tour": places_list_tour,
        }
    )


@tool
def remove_place_from_tour(
    place_display_name: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    state: Annotated[dict, InjectedState],
    config: RunnableConfig,
) -> Command:
    """
    Remove a place from the tour list.

    Args:
        place_display_name: Display Name of the place to remove from the tour

    Returns:
        Updated tour list
    """
    try:
        tour = FieldGetter(state).get_field("tour", list[str])
    except KeyError as e:
        raise ValueError("The tour is still empty") from e

    places_list = FieldGetter(state).get_field(
        "places_list", PlacesList, raise_error_if_missing=False
    )
    if places_list is None:
        places_list = PlacesList()

    try:
        tour.remove(place_display_name)
        places_list_tour = places_list.get_by_display_names(tour)
    except ValueError as e:
        raise ValueError(
            f"Place {place_display_name} not found in the tour list. Actual values in the tour list: {tour} "
            "If you got this error, check the spelling of the display name, but if the place is not it here "
            "at all it means that the place you are trying to remove is not in the tour list. "
            "In that case, just go on with the next step."
        ) from e

    return Command(
        update={
            "messages": [
                ToolMessage(
                    content=f"Removed {place_display_name} from the tour list\n\nThis is the updated tour:\n{tour}",
                    tool_call_id=tool_call_id,
                )
            ],
            "tour": tour,
            "places_list_tour": places_list_tour,
        }
    )


tools = [place_search, places_nearby, add_place_to_tour, remove_place_from_tour]


llm = ChatVertexAI(
    model=LLM,
    temperature=0.2,
    max_tokens=4096,
    streaming=True,
).bind_tools(tools)

# checkpointer = MemorySaver()

agent = create_react_agent(
    llm,
    tools,
    state_schema=AgentState,
    prompt=SYSTEM_PROMPT,
    # checkpointer=checkpointer,
)

if __name__ == "__main__":

    def print_stream(stream):
        for s in stream:
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

    while True:
        user_message = input("Enter your message: ")
        print_stream(
            agent.stream(
                input={
                    "messages": [
                        (
                            "user",
                            user_message,
                        )
                    ]
                },
                config=RunnableConfig(configurable={"thread_id": "2"}),
                stream_mode="values",
            )
        )
