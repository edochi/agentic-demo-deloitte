"""
Demo application to showcase places with audio files in Streamlit.
Run this with: streamlit run frontend/demo_places_audio.py
"""

import streamlit as st

from frontend.utils.sample_places_data import get_sample_places


def main():
    """Main demo application function."""
    st.set_page_config(
        page_title="Places with Audio Demo",
        layout="wide",
        initial_sidebar_state="auto",
    )

    st.title("Places with Audio Demo")
    st.markdown("""
    This demo showcases the ability to display places with descriptions and audio files.
    Click on each place to expand and listen to the audio.
    """)

    # Get sample places data
    sample_data = get_sample_places()

    # Display places with audio
    st.write("### Places of Interest")

    for i, place in enumerate(sample_data["places"]):
        with st.expander(f"{place.get('name', f'Place {i + 1}')}"):
            st.write(place.get("description", "No description available"))

            if "audio_file" in place:
                audio_file = place["audio_file"]
                try:
                    st.audio(audio_file)
                except Exception as e:
                    st.error(f"Error playing audio file: {e}")
            else:
                st.write("No audio file available for this place.")

    # Show how to use this in a chat application
    st.subheader("Integration with Chat Application")
    st.markdown("""
    In a chat application, you can send a message like this to get information about places:
    
    ```
    Tell me about famous landmarks in Europe
    ```
    
    The AI response will include a document object with places and audio files that can be played directly in the chat interface.
    """)

    # Display sample code
    st.subheader("Sample Code for Handling Places with Audio in Chat")
    st.code(
        """
# Example response format
response = {
    "content": {
        "places": [
            {
                "name": "Eiffel Tower",
                "description": "The Eiffel Tower is a wrought-iron lattice tower...",
                "audio_file": "https://example.com/audio/eiffel_tower.mp3"
            },
            {
                "name": "Colosseum",
                "description": "The Colosseum is an oval amphitheatre...",
                "audio_file": "https://example.com/audio/colosseum.mp3"
            }
        ]
    },
    "type": "ai"
}

# In your display function
if message["type"] == "ai" and isinstance(message["content"], dict) and "places" in message["content"]:
    display_places_with_audio(message["content"]["places"])
else:
    st.markdown(format_content(message["content"]), unsafe_allow_html=True)
    """,
        language="python",
    )


if __name__ == "__main__":
    main()
