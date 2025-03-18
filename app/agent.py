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
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
import wikipedia
import requests
import base64

LOCATION = "us-central1"
LLM = "gemini-2.0-flash-001"

# 1. Define tools
@tool
def search_wiki_page(topic: str) -> list[str]:
    """Search on wikipedia for a specific page. Use it to get the name of the page you want to analyze"""
    search = wikipedia.search(topic)
    return search

@tool
def get_wiki_information(page: str) -> str:
    """Get a summary of a wikipedia page. Use it to get the summarization of the page you want to analyze"""
    summary = wikipedia.summary(page)
    LOCATION = "us-central1"
    LLM = "gemini-2.0-flash-001"
    llm = ChatVertexAI(model=LLM, location=LOCATION, temperature=0, max_tokens=1024, project='qwiklabs-gcp-02-44d130f8f4a0')
    messages = [
        (
            "system",
            "You have to create a summary of the text provided by the user. Your summary will be used by a tour guide. Return only the summary without any comment or explanation.",
        ),
        ("human", summary)
        ]
    agent_summary=llm.invoke(messages).content
    return agent_summary

tools = [search_wiki_page, get_wiki_information]

# 2. Set up the language model
llm = ChatVertexAI(
    model=LLM, location=LOCATION, temperature=0, max_tokens=1024, streaming=True
).bind_tools(tools)


# 3. Define workflow components
def should_continue(state: MessagesState) -> str:
    """Determines whether to use tools or end the conversation."""
    last_message = state["messages"][-1]
    return "tools" if last_message.tool_calls else END


def call_model(state: MessagesState, config: RunnableConfig) -> dict[str, BaseMessage]:
    """Calls the language model and returns the response."""
    system_message = "You are a helpful AI assistant that returns summarization of some topics using your tools to retrieve data from wikipedia."
    messages_with_system = [{"type": "system", "content": system_message}] + state[
        "messages"
    ]
    # Forward the RunnableConfig object to ensure the agent is capable of streaming the response.
    response = llm.invoke(messages_with_system, config)
    return {"messages": response}


# 4. Create the workflow graph
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.set_entry_point("agent")

# 5. Define graph edges
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")

# 6. Compile the workflow
agent = workflow.compile()

"""
json={
  "input": {
    "text": agent_summary
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
headers={"Authorization": "Bearer ya29.a0AeXRPp44IPsKCtIN2QIjyAY8GOBubN-LhoPJK79HveL70LzTVBJhq2BJwL41T5GxhMUpQS4OR44VXy_KteRid68_C1MFABjbe_cDTIOM-Ha6x89WxrX5eKSF7cYvEb4njq_SjbLiJ7zkvhJFjzXkaWZqp_BkJRRSW-eiJ28ORMUPB1HL78rt0mnWtc4yc4CWD_CYeEXpxD4RtkpB7ymwsd-uTIrkU26TuoMbXfWGb-9DMVdxsQ5Myrp_PQdzLDnePyNP53ez5FWx3GvWfk5vpg924Wy9rmZreLuPYRXRgBYJGhbwACGc_zguQlJAQBzylpwtIv9SN1PdZ7X6IGmwbRlWF9YxxoHO3Pl2kkycMV9oPqjY1L9elrrrs47aLDSx5thwb6OkMwGBrnPcLk-q5Izdnx8V8lPzHJZnaQaCgYKAUQSARISFQHGX2Mi5wlxcSQmjYzX94Xb7jqFLQ0429",
        "x-goog-user-project": "qwiklabs-gcp-02-44d130f8f4a0",
        "Content-Type": "application/json; charset=utf-8"}
data = requests.post('https://texttospeech.googleapis.com/v1/text:synthesize',json=json, headers=headers)
print(data.json())
decoded_data = base64.b64decode(data.json()['audioContent'])
with open ('test_audio.mp3', 'wb') as f_audio:
    f_audio.write(decoded_data)
"""