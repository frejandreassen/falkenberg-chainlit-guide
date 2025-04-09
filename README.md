# Falkenberg Tourism Assistant

A Chainlit application that serves as a tourism assistant for Falkenberg municipality. The assistant leverages both OpenAI's GPT-4o and Google's Gemini to provide comprehensive information about attractions, activities, accommodations, restaurants, and events in Falkenberg.

## Features

- Complete tourism information using Gemini's large context window
- Comprehensive event information via direct CMS GraphQL API integration
- Structured event data following specific formatting requirements
- Automatic data refresh from CMS every 3 hours
- Bilingual support (Swedish and English)
- Streaming responses for a better user experience

## Architecture

- **Frontend**: Chainlit chat interface
- **Primary LLM**: OpenAI GPT-4o for conversation handling and response formatting
- **Knowledge Backend**: Google Gemini 1.5 Pro for processing both event and tourism data
- **Data Source**: Direct integration with Falkenberg CMS via GraphQL API

## Setup Instructions

1. **Clone the repository**

```bash
git clone <repository-url>
cd falkenberg-tourism-assistant
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Set up environment variables**

Copy the `.env.template` file to `.env` and fill in your API keys:

```bash
cp .env.template .env
```

Edit the `.env` file and add your API keys:
- `OPENAI_API_KEY`: Your OpenAI API key
- `GOOGLE_API_KEY`: Your Google API key for Gemini

4. **Run the application**

```bash
chainlit run app.py
```

## Project Structure

- `app.py`: Main Chainlit application
- `gemini_tools.py`: Contains the GeminiTools class for Gemini integration with both events and tourism data
- `.env`: Environment variables (API keys)
- `requirements.txt`: Python dependencies

## Usage

The assistant can answer questions about:

1. **Tourism information**: Attractions, accommodations, restaurants, etc.
   ```
   What are the best beaches in Falkenberg?
   ```

2. **Event-related information**: Upcoming events, festivals, activities
   ```
   Are there any music festivals in Falkenberg this summer?
   ```

The assistant automatically detects the query type and uses the appropriate Gemini function to provide the most accurate information.

## Dependencies

- Python 3.8+
- chainlit
- openai
- google-generativeai
- python-dotenv
- requests

## CMS Integration

The application fetches both event and tourism data directly from the Falkenberg CMS using GraphQL. The queries are structured based on the schema available at `https://cms.falkenberg.se/graphql`.

### Gemini System Prompts

#### Events Prompt
The Gemini events assistant is configured as a knowledge expert that follows a strict output format:

```
**Evenemang:**
***[Evenemangets namn]:**
* Beskrivning: [Description]
* Datum och tid: [Date and time]
* Plats: [Location]
* URI: [URI]
```

#### Tourism Prompt
The Gemini tourism assistant is configured as a knowledgeable tour guide providing comprehensive information about Falkenberg's attractions, accommodations, and activities.

OpenAI GPT-4o then transforms these structured responses into natural, conversational answers for the end user.