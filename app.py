import os
import chainlit as cl
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Import GeminiTools class
from gemini_tools import GeminiTools

# Initialize GeminiTools
gemini_tools = GeminiTools(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    cms_url="https://cms.falkenberg.se/graphql"
)

# Schedule regular refresh of events data (every 3 hours)
gemini_tools.schedule_refresh(interval_hours=3)

# Define function schemas for the tools
function_schemas = [
    {
        "type": "function",
        "function": {
            "name": "ask_gemini_about_events",
            "description": "Ask Gemini about events in Falkenberg. Use this for any event-related questions including activities, concerts, festivals, and local happenings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The event-related query or question about Falkenberg events"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_gemini_about_pages",
            "description": "Ask Gemini about information on the Falkenberg tourism website. Use this for general tourism questions, attractions, services, and information about Falkenberg that would be found on their website.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The website-related query or question about Falkenberg tourism"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Map function names to their implementations
available_functions = {
    "ask_gemini_about_events": gemini_tools.ask_gemini_about_events,
    "ask_gemini_about_pages": gemini_tools.ask_gemini_about_pages
}

# Process query to be more specific about dates
def process_date_references(query):
    # Get next weekend dates
    today = datetime.now()
    days_until_saturday = (5 - today.weekday()) % 7
    next_saturday = today + timedelta(days=days_until_saturday)
    next_sunday = next_saturday + timedelta(days=1)
    next_weekend = f"{next_saturday.strftime('%Y-%m-%d')} to {next_sunday.strftime('%Y-%m-%d')}"
    
    # Check for weekend references and make them explicit
    if any(term in query.lower() for term in ["helgen", "weekend", "helg", "veckoslut"]):
        # Append weekend date information to the query
        return f"{query} (referring to dates {next_weekend})"
    return query

@cl.on_chat_start
async def start_chat():
    # Check if we have any events and pages data
    events_data = gemini_tools.get_events_data()
    pages_data = gemini_tools.get_pages_data()
    events_status = "active" if events_data else "unavailable"
    pages_status = "active" if pages_data else "unavailable"
    
    # Get today's date with weekday
    current_date = datetime.now().strftime("%A, %Y-%m-%d")
    
    # Get next weekend dates
    today = datetime.now()
    days_until_saturday = (5 - today.weekday()) % 7
    next_saturday = today + timedelta(days=days_until_saturday)
    next_sunday = next_saturday + timedelta(days=1)
    next_weekend = f"{next_saturday.strftime('%Y-%m-%d')} to {next_sunday.strftime('%Y-%m-%d')}"
    
    # Set up the message history with system message
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": f"""You are a helpful tourism assistant for Falkenbergs kommun. You are a chatbot on falkenberg.se. You help visitors find information about Falkenberg.

For ANY question about Falkenberg, including events, activities, attractions, restaurants, accommodations, or general information:
1. ALWAYS use one of the available tools (ask_gemini_about_events or ask_gemini_about_pages).
2. NEVER claim you don't have information without first checking with one of the tools.
3. For event-related questions (festivals, concerts, activities, happenings), use the ask_gemini_about_events tool.
4. For all other questions (restaurants, attractions, accommodations, general tourism, etc.), use the ask_gemini_about_pages tool.

Current data status:
- Events data: {events_status}
- Website data: {pages_status}
- Today's date: {current_date}
- This weekend refers to: {next_weekend}

DATE HANDLING INSTRUCTIONS:
- When a user asks about "helgen" or "this weekend", ALWAYS use the date range {next_weekend}
- Format date references clearly in your tool queries
- Be precise about date ranges when asking for event information

When asked any question about local services, places, or recommendations in Falkenberg, ALWAYS use the ask_gemini_about_pages tool first before responding. This includes restaurants, shops, attractions, beaches, or any other local information.

Always try to provide helpful, specific answers that would be useful to tourists visiting Falkenberg. Your base URL is https://falkenberg.se.

You can respond in either Swedish or English, or any other language matching the language used by the visitor. If the user writes in Swedish, answer in Swedish. If the user writes in English, answer in English. If the user writes in German, answer in German"""}]
    )
    # Welcome message with multilingual greeting
    welcome_message = """🇸🇪 Välkommen till Falkenberg! 
🇬🇧 Welcome to Falkenberg!
🇩🇪 Willkommen in Falkenberg!

Hur kan jag hjälpa dig idag? / How can I help you today? / Wie kann ich Ihnen heute helfen?"""

    # Welcome message
    await cl.Message(content=welcome_message).send()

@cl.on_message
async def main(message: cl.Message):
    # Get message history from session
    message_history = cl.user_session.get("message_history")
    
    # Add user message to history
    message_history.append({"role": "user", "content": message.content})
    
    # Create empty message for streaming
    msg = cl.Message(content="")
    await msg.send()
    
    # First call to get either a direct response or tool calls
    try:
        stream = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=message_history,
            tools=function_schemas,
            tool_choice="auto",
            stream=True
        )
        
        response_text = ""
        tool_calls = []
        
        async for chunk in stream:
            delta = chunk.choices[0].delta
            
            if delta.content:
                await msg.stream_token(delta.content)
                response_text += delta.content
                
            elif delta.tool_calls:
                tcchunks = delta.tool_calls
                for tcchunk in tcchunks:
                    if len(tool_calls) <= tcchunk.index:
                        tool_calls.append({"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    tc = tool_calls[tcchunk.index]
                    
                    if tcchunk.id:
                        tc["id"] += tcchunk.id
                    if tcchunk.function and tcchunk.function.name:
                        tc["function"]["name"] += tcchunk.function.name
                    if tcchunk.function and tcchunk.function.arguments:
                        tc["function"]["arguments"] += tcchunk.function.arguments
        
        # If we got a direct text response
        if response_text:
            message_history.append({"role": "assistant", "content": response_text})
            # Properly finish streaming without the pulsating dot
            await msg.update()
        
        # If we need to call tools
        if tool_calls:
            # Add the tool calls to the message history
            message_history.append({"role": "assistant", "tool_calls": tool_calls})
            
            # Display temporary loading text with ellipsis
            loading_text = "Söker information"  # "Searching for information" in Swedish
            await msg.stream_token(loading_text)
            
            # Process each tool call
            for tool_call in tool_calls:
                function_name = tool_call["function"]["name"]
                
                # Show loading animation by adding dots
                await msg.stream_token(".")
                
                # Only process functions we know about
                if function_name in available_functions:
                    function_to_call = available_functions[function_name]
                    try:
                        # Parse arguments and call function
                        function_args = json.loads(tool_call["function"]["arguments"])
                        
                        # Process the query to be more explicit about dates
                        if "query" in function_args:
                            function_args["query"] = process_date_references(function_args["query"])
                        
                        # Add another dot for loading animation
                        await msg.stream_token(".")
                        
                        function_response = function_to_call(**function_args)
                        
                        # Add function response to message history
                        message_history.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": function_response
                        })
                        
                        # Update the data status in the system message if needed
                        if "Sorry, I couldn't retrieve any event data" in function_response:
                            # Update the system message with current data status
                            system_msg = message_history[0]["content"]
                            updated_system_msg = system_msg.replace("Events data: active", 
                                                                     "Events data: unavailable")
                            message_history[0]["content"] = updated_system_msg
                        
                        if "Sorry, I couldn't retrieve any page data" in function_response:
                            # Update the system message with current data status
                            system_msg = message_history[0]["content"]
                            updated_system_msg = system_msg.replace("Website data: active", 
                                                                     "Website data: unavailable")
                            message_history[0]["content"] = updated_system_msg
                        
                    except json.JSONDecodeError as e:
                        print(f"Error parsing function arguments: {e}")
                        continue
                    except Exception as e:
                        error_message = f"Error calling function {function_name}: {str(e)}"
                        print(error_message)
                        message_history.append({
                            "tool_call_id": tool_call["id"],
                            "role": "tool",
                            "name": function_name,
                            "content": error_message
                        })
            
            # Make a second call to process the tool results
            second_stream = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=message_history,
                stream=True
            )
            
            # Reset content by creating a new message
            # msg = cl.Message(content="")
            msg.content=""
            await msg.send()
            
            # Stream the final response
            final_response = ""
            async for chunk in second_stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    await msg.stream_token(token)
                    final_response += token
            
            # Add final response to message history
            if final_response:
                message_history.append({"role": "assistant", "content": final_response})
            
            # IMPORTANT: Properly finish streaming to remove the pulsating dot
            await msg.update()
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        # Create a new message with the error instead of updating
        error_msg = cl.Message(content=error_message)
        await error_msg.send()
        print(error_message)
    
    # Update the message history in session
    cl.user_session.set("message_history", message_history)

if __name__ == "__main__":
    cl.run()