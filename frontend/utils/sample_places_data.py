"""
Sample data for places with audio files.
This module provides sample data to demonstrate the places with audio feature.
"""

SAMPLE_PLACES = {
    "places": [
        {
            "name": "Eiffel Tower",
            "description": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
            "audio_file": "https://upload.wikimedia.org/wikipedia/commons/7/7b/En-us-Eiffel_Tower.ogg"
        },
        {
            "name": "Colosseum",
            "description": "The Colosseum is an oval amphitheatre in the centre of the city of Rome, Italy. It is the largest ancient amphitheatre ever built, and is still the largest standing amphitheatre in the world today.",
            "audio_file": "https://upload.wikimedia.org/wikipedia/commons/2/29/En-us-colosseum.ogg"
        },
        {
            "name": "Taj Mahal",
            "description": "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in the Indian city of Agra. It was commissioned in 1632 by the Mughal emperor Shah Jahan to house the tomb of his favourite wife, Mumtaz Mahal.",
            "audio_file": "https://upload.wikimedia.org/wikipedia/commons/4/4a/En-us-taj_mahal.ogg"
        }
    ]
}

def get_sample_places():
    """Returns the sample places data."""
    return SAMPLE_PLACES
