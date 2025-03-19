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
    model=LLM, 
    location=LOCATION, 
    temperature=0, 
    max_tokens=1024, 
    project=PROJECT
).bind_tools(tools)

# 4. Define workflow components
def should_continue(state: WikiTourState) -> str:
    """Determines whether to use tools or end the conversation."""
    last_message = state["messages"][-1]
    
    # Se summary è stata impostata a stringa vuota (errore), termina
    if state.get("summary") == "":
        return END
    
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
    """
    
    messages_with_system = [{"type": "system", "content": system_message}] + state["messages"]
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
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        tool_call_id = tool_call['id']
        
        if tool_name == "search_wiki_page":
            result = search_wiki_page.invoke(tool_args["topic"])
            new_state["wiki_pages"] = result
            tool_results.append(ToolMessage(
                name=tool_name, 
                content=str(result), 
                tool_call_id=tool_call_id
            ))
            
        elif tool_name == "get_wiki_content":
            page = tool_args["page"]
            new_state["selected_page"] = page
            result = get_wiki_content.invoke(page)
            new_state["wiki_content"] = result
            
            # Verifica se il risultato contiene un errore
            if result.startswith("Error retrieving Wikipedia page"):
                # Imposta summary come stringa vuota
                new_state["summary"] = ""
                # Aggiungi il messaggio di errore come tool result
                tool_results.append(ToolMessage(
                    name=tool_name, 
                    content=result, 
                    tool_call_id=tool_call_id
                ))
                # Aggiungi alla lista dei messaggi
                new_state["messages"] = messages + tool_results
                # Ritorna lo stato così com'è per terminare il flusso
                return new_state
            
            tool_results.append(ToolMessage(
                name=tool_name, 
                content=result, 
                tool_call_id=tool_call_id
            ))
    
    new_state["messages"] = messages + tool_results
    return new_state

def initialize_agent(topic: str) -> WikiTourState:
    """Initialize the agent with a custom state."""
    initial_state = {
        "messages": [HumanMessage(content=f"Research the topic: {topic}")],
        "wiki_pages": [],
        "selected_page": None,
        "wiki_content": None,
        "summary": None
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