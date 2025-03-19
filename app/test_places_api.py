import json
from enum import StrEnum

import requests
from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator


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
    cafe = "cafe"
    cafeteria = "cafeteria"
    fast_food_restaurant = "fast_food_restaurant"
    fine_dining_restaurant = "fine_dining_restaurant"
    pizza_restaurant = "pizza_restaurant"
    pub = "pub"
    restaurant = "restaurant"
    clothing_store = "clothing_store"
    gift_shop = "gift_shop"
    jewelry_store = "jewelry_store"
    market = "market"
    shoe_store = "shoe_store"
    shopping_mall = "shopping_mall"
    sporting_goods_store = "sporting_goods_store"
    store = "store"


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
    text: LocalizedText | None = None
    original_text: LocalizedText | None = Field(None, alias="originalText")
    rating: float

    model_config = ConfigDict(extra="ignore")


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
    website_uri: str | None = Field(None, alias="websiteUri")
    user_rating_count: int | None = Field(None, alias="userRatingCount")
    display_name: DisplayName = Field(alias="displayName")
    reviews: list[Review] | None = None
    generative_summary: str | None = Field(None, alias="generativeSummary")

    model_config = ConfigDict(extra="ignore")

    def get_nearby_locations(
        self, radius: float, location_types: list[PlaceTypes] | None = None
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
            "places.internationalPhoneNumber,"
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
            "maxResultCount": 20,
        }
        if location_types is not None:
            data["includedTypes"] = location_types

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            return PlacesList.model_validate(response.json())

        except requests.exceptions.RequestException as e:
            raise Exception(f"Error while searching for nearby locations: {e}") from e


class PlacesList(BaseModel):
    places: list[Place]

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

    def append(self, place: Place) -> None:
        self.places.append(place)

    def extend(self, places: "PlacesList") -> None:
        self.places.extend(places.places)

    def get_by_display_name(self, display_name: str) -> Place:
        try:
            return next(
                place
                for place in self.places
                if place.display_name.text == display_name
            )
        except StopIteration as e:
            raise IndexError(
                f"Place with display name {display_name} not found: {e}"
            ) from e
        except Exception as e:
            raise Exception(
                f"Error while searching for place with display name {display_name}: {e}"
            ) from e

    def get_by_display_names(self, display_names: list[str]) -> list[Place]:
        return [
            self.get_by_display_name(display_name) for display_name in display_names
        ]


class PlaceSimplified(BaseModel):
    display_name: str
    address: str
    rating: float | None = None
    website: str | None = None
    phone_number: str | None = None
    types: list[str] | None = None

    @classmethod
    def from_place(cls, place: Place) -> "PlaceSimplified":
        return cls(
            display_name=place.display_name.text,
            address=place.formatted_address,
            rating=place.rating,
            website=place.website_uri,
            phone_number=place.international_phone_number,
        )

    def get_structured_string(self) -> str:
        rating = f"Rating: {self.rating}\n" if self.rating is not None else ""
        website = f"Website: {self.website}\n" if self.website is not None else ""
        phone_number = (
            f"Phone number: {self.phone_number}\n"
            if self.phone_number is not None
            else ""
        )
        types = f"Types: {', '.join(self.types)}\n" if self.types is not None else ""
        return (
            f"Display Name: {self.display_name}\n"
            f"Address: {self.address}\n"
            f"{rating}"
            f"{website}"
            f"{phone_number}"
            f"{types}"
        )


class PlacesListSimplified(BaseModel):
    places: list[PlaceSimplified]

    @classmethod
    def from_places_list(cls, places_list: PlacesList) -> "PlacesListSimplified":
        return cls(
            places=[PlaceSimplified.from_place(place) for place in places_list.places]
        )

    def get_structured_string(self) -> str:
        return "\n\n".join(place.get_structured_string() for place in self.places)


class Route(BaseModel):
    distance_meters: int = Field(alias="distanceMeters")
    duration: int

    @field_validator("duration", mode="before")
    @classmethod
    def validate_duration(cls, value: str) -> int:
        if not value.endswith("s"):
            raise ValueError("Duration must be a string ending with 's'")
        return int(value.replace("s", ""))

    @computed_field
    @property
    def humanized_distance(self) -> str:
        return f"{self.distance_meters / 1000} km"

    @computed_field
    @property
    def humanized_duration(self) -> str:
        hours = self.duration // 3600
        remaining_after_hours = self.duration % 3600
        minutes = remaining_after_hours // 60
        seconds = remaining_after_hours % 60

        return f"{hours} hours, {minutes} minutes, {seconds} seconds"


class Routes(BaseModel):
    routes: list[Route]


def get_places_distance(origin: Place, destination: Place) -> Routes:
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": "AIzaSyDjoRzcHj72yIdFPSLTr4bJ5ywR7ltwVXY",  # Sostituisci con la tua API key
        # 'X-Goog-FieldMask': 'routes.duration,routes.distanceMeters,routes.polyline.encodedPolyline'
        "X-Goog-FieldMask": "routes.distanceMeters,routes.duration",
    }

    payload = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": origin.location.latitude,
                    "longitude": origin.location.longitude,
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": destination.location.latitude,
                    "longitude": destination.location.longitude,
                }
            }
        },
        "travelMode": "WALK",
        # "routingPreference": "TRAFFIC_UNAWARE",
        "computeAlternativeRoutes": False,
        "routeModifiers": {
            "avoidTolls": False,
            "avoidHighways": False,
            "avoidFerries": False,
        },
        "languageCode": "en-US",
        "units": "METRIC",
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Solleva un'eccezione per codici di stato HTTP non riusciti (4xx o 5xx)
        return Routes.model_validate(response.json())
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error while computing routes: {e}") from e


if __name__ == "__main__":
    # Cerca un luogo dato un nome/descrizione
    # place_info = text_search("Rome, Italy")
    # print(json.dumps(place_info, indent=2))

    # print(text_search("colosseo, roma"))

    # locations = nearby_search(
    #     41.890251, 12.492373, 1000, [PlaceTypes.tourist_attraction]
    # )

    # print(locations.get_by_display_name("Roman Forum"))

    # print(text_search("colosseo, roma"))

    place1 = PlacesList.search_places("castello del valentino, torino").places[0]
    place2 = PlacesList.search_places("castello sforzesco, milano").places[0]

    print(get_places_distance(place1, place2))
