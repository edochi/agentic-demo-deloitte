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
import base64
from typing import List, Optional, TypedDict

import requests
import wikipedia
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"
PROJECT = "qwiklabs-gcp-02-44d130f8f4a0"


# 1. Define custom state
class WikiTourState(TypedDict):
    """Custom state for the Wikipedia tour guide agent."""

    messages: List[BaseMessage]
    wiki_pages: List[str]
    selected_page: Optional[str]
    wiki_content: Optional[str]
    summary: Optional[str]


# 2. Define tools
@tool
def search_wiki_page(topic: str) -> list[str]:
    """Search on wikipedia for a specific page. Use it to get the name of the page you want to analyze"""
    search = wikipedia.search(topic)
    print(search)
    return search


@tool
def get_wiki_content(page: str) -> str:
    """Get the full content of a wikipedia page."""
    try:
        summary = wikipedia.summary(page)
        return summary
    except Exception as e:
        return f"Error retrieving Wikipedia page '{page}': {str(e)}"


tools = [search_wiki_page, get_wiki_content]

# 3. Set up the agent model with tools
llm = ChatVertexAI(
    model=LLM, location=LOCATION, temperature=0, max_tokens=1024, project=PROJECT
).bind_tools(tools)


# 4. Define workflow components
def should_continue(state: WikiTourState) -> str:
    """Determines whether to use tools or end the conversation."""
    last_message = state["messages"][-1]

    # Check if this is the final AI message with a summary
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        # Update the summary in the state
        state["summary"] = last_message.content
        return END

    return "tools" if last_message.tool_calls else END


def call_model(state: WikiTourState, config: RunnableConfig) -> WikiTourState:
    """Calls the language model and returns the response."""
    system_message = """
        You are an autonomous agent that researches a topic on Wikipedia and creates a concise summary suitable for a tour guide.

        Follow these steps IN ORDER:
        1. FIRST, use search_wiki_page to find relevant Wikipedia pages about the topic
        2. THEN, select the most appropriate page from the list returned by the tool
        3. NEXT, use get_wiki_content to retrieve the actual content of the most appropriate page
        4. FINALLY, create a concise, well-structured summary of the information suitable for a tour guide

        INSTRUCTIONS:
        - Your research process MUST use the tools provided. Do not skip using the tools.
        - Only after you have collected information using the tools should you create your final summary.
        - Focus on interesting facts, historical significance, and engaging details that would interest tourists. Keep it short.
        - After using the tools, return only the final summary, without including any additional comment or introduction.
        - If you encounter an error after a tool call, then the final summary you return must be a blank string (""), without any additional comment or introduction.
    """

    messages_with_system = [{"type": "system", "content": system_message}] + state[
        "messages"
    ]
    response = llm.invoke(messages_with_system, config)

    # Update state with new message
    new_state = state.copy()
    new_state["messages"] = state["messages"] + [response]

    # If this is the final response, save it as the summary
    if not response.tool_calls:
        new_state["summary"] = response.content

    return new_state


def process_tool_calls(state: WikiTourState) -> WikiTourState:
    """Process tool calls and update the custom state with relevant information."""
    new_state = state.copy()
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return new_state

    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_call_id = tool_call["id"]

        if tool_name == "search_wiki_page":
            result = search_wiki_page.invoke(tool_args["topic"])
            new_state["wiki_pages"] = result
            tool_results.append(
                ToolMessage(
                    name=tool_name, content=str(result), tool_call_id=tool_call_id
                )
            )

        elif tool_name == "get_wiki_content":
            page = tool_args["page"]
            new_state["selected_page"] = page
            result = get_wiki_content.invoke(page)
            new_state["wiki_content"] = result
            tool_results.append(
                ToolMessage(name=tool_name, content=result, tool_call_id=tool_call_id)
            )

    new_state["messages"] = messages + tool_results
    return new_state


def initialize_agent(topic: str) -> WikiTourState:
    """Initialize the agent with a custom state."""
    initial_state = {
        "messages": [HumanMessage(content=f"Research the topic: {topic}")],
        "wiki_pages": [],
        "selected_page": None,
        "wiki_content": None,
        "summary": None,
    }
    return initial_state


# 5. Create the workflow graph with custom state
workflow = StateGraph(WikiTourState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", process_tool_calls)
workflow.set_entry_point("agent")

# 6. Define graph edges
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 7. Compile the workflow
autonomous_agent = workflow.compile()


# 8. Function to run the autonomous agent and return the complete custom state
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


def get_landmark_description(topic: str):
    summary = get_tour_guide_summary(topic)["summary"]
    json = {
        "input": {"text": summary},
        "voice": {
            "languageCode": "en-gb",
            "name": "en-GB-Standard-A",
            "ssmlGender": "FEMALE",
        },
        "audioConfig": {"audioEncoding": "MP3"},
    }
    headers = {
        "Authorization": "Bearer ya29.a0AeXRPp5urN8iUFropcX02yVgpmiNMWERUIxeXeV9GJ0MsRdK97wCq6rC2UhAWTcBlGwhpF0K-dM_0bHgVt_6ifir8rLHUw7QHf17IWSQnQeYoAiyBZEFViK1e0CyYml91N8F5brlvelvzjVhgOfY0JCI_oZb6KwjS_7IAdcMBF8LizOm9idvzgr3EfCwoB51ZnWpTo08BPL_h60mTIQiXtTXvRVuciLGVhehAgtO46zYj_dx0jCnrfD9sP7szm3NEkLb3pvuz8jGhPYvs3O460VQWFvnsXmx8cIZmNSP4o7ylNhRJ447Bzw1jMEzVTnT0nBanbZgawE0GpPYb8YkM7AkdvofAWzp6eDB_hCOU6JOA8S4JCTTfJ5cHFb-NXe38OLw5huWK4LEs2PFphoys5n_xo1ovxGKjCtiJUsaCgYKAUISARISFQHGX2MiOPcBMmL18lzkjXuWMqtj6g0430",
        "x-goog-user-project": PROJECT,
        "Content-Type": "application/json; charset=utf-8",
    }
    data = requests.post(
        "https://texttospeech.googleapis.com/v1/text:synthesize",
        json=json,
        headers=headers,
    )
    decoded_data = base64.b64decode(data.json()["audioContent"])
    with open("test_audio.mp3", "wb") as f_audio:
        f_audio.write(decoded_data)
    return


get_landmark_description("Fontana di Trevi")
