import os
import json
import time
import re
import html
import requests
import tiktoken
from bs4 import BeautifulSoup
from google.generativeai import GenerativeModel
import google.generativeai as genai
from typing import List, Dict, Any, Optional

class GeminiTools:
    def __init__(self, google_api_key: str, cms_url: str = "https://cms.falkenberg.se/graphql", log_file: str = "gemini_log.txt"):
        self.cms_url = cms_url
        self.google_api_key = google_api_key
        self.log_file = log_file
        
        # Configure Gemini API
        genai.configure(api_key=google_api_key)
        self.model = GenerativeModel(
            model_name="gemini-2.0-flash",
            generation_config={
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
            }
        )
        
        # Cache setup for events
        self.events_cache = {"data": None, "last_updated": 0, "cache_duration": 1800}
        
        # Cache setup for pages
        self.pages_cache = {"data": None, "last_updated": 0, "cache_duration": 3600}
        
        # Setup tokenizer
        try:
            # Use cl100k_base tokenizer (used by GPT-4 models)
            # This is a good approximation for Gemini
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self._log("SYSTEM", "Initialized tokenizer")
        except Exception as e:
            self._log("ERROR", f"Failed to initialize tokenizer: {str(e)}")
            self.tokenizer = None
            
        # Initialize log file
        self._log("SYSTEM", "Initialized GeminiTools")
        
        # Load event data
        self.refresh_events_data()
        
        # Load pages data
        self.refresh_pages_data()
    
    def _log(self, source: str, message: str):
        """Simple logging to file"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"[{timestamp}] {source}: {message}\n")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string"""
        if not self.tokenizer:
            return 0
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            self._log("ERROR", f"Token counting error: {str(e)}")
            return 0
    
    # EVENTS FUNCTIONS
    
    def fetch_events_data(self) -> List[Dict[str, Any]]:
        """Fetch events data from CMS GraphQL API"""
        query = """
        query AllEvent {
          allEvent(first: 10000) {
            nodes {            
                content
                location {
                    active
                    name
                }
                slug
                title
                uri
                acfGroupEvent {
                    bookingLink
                    occasions {
                        startDate
                        endDate
                    }
                    rcrRules {
                        rcrStartDate
                        rcrEndDate
                        rcrStartTime
                        rcrEndTime
                        rcrWeekDay
                        rcrWeeklyInterval
                        rcrExceptions {
                            rcrExcDate
                        }
                    }
                }
                date
            }
          }
        }
        """
        
        try:
            response = requests.post(
                self.cms_url,
                json={"query": query},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                nodes = data.get("data", {}).get("allEvent", {}).get("nodes", [])
                return nodes
            else:
                self._log("ERROR", f"Failed to fetch events: HTTP {response.status_code}")
                return []
        except Exception as e:
            self._log("ERROR", f"Error fetching events: {str(e)}")
            return []
    
    def get_events_data(self) -> List[Dict[str, Any]]:
        """Get events data (from cache if valid)"""
        current_time = time.time()
        
        if (self.events_cache["data"] is None or 
            current_time - self.events_cache["last_updated"] > self.events_cache["cache_duration"]):
            self.refresh_events_data()
        
        return self.events_cache["data"] or []
    
    def refresh_events_data(self) -> None:
        """Refresh the events data and update cache"""
        fresh_data = self.fetch_events_data()
        self.events_cache["data"] = fresh_data
        self.events_cache["last_updated"] = time.time()
        self._log("SYSTEM", f"Refreshed events data. Total events: {len(fresh_data)}")
    
    def format_dates(self, occasions, rcr_rules=None):
        """Format dates concisely, handling both individual occasions and recurring events"""
        formatted = []
        
        # Process individual occasions
        if occasions:
            for occ in occasions or []:
                start = occ.get('startDate', '')
                end = occ.get('endDate', '')
                
                if start and end and start == end:
                    formatted.append(start)
                elif start and end:
                    formatted.append(f"{start} to {end}")
                elif start:
                    formatted.append(start)
        
        # Process recurring rules if available
        if rcr_rules:
            for rule in rcr_rules:
                weekday = rule.get('rcrWeekDay', '')
                start_date = rule.get('rcrStartDate', '')
                end_date = rule.get('rcrEndDate', '')
                start_time = rule.get('rcrStartTime', '')
                interval = rule.get('rcrWeeklyInterval', 1)
                
                if weekday and start_date and end_date:
                    # Format: "Every Tuesday at 18:00, July 1 - August 12, 2025"
                    date_range = f"{start_date.split('-')[1]}/{start_date.split('-')[2]}" if start_date else ""
                    date_range += f" - {end_date.split('-')[1]}/{end_date.split('-')[2]}" if end_date else ""
                    date_range += f", {start_date.split('-')[0]}" if start_date else ""
                    
                    time_info = f" at {start_time}" if start_time else ""
                    
                    interval_text = ""
                    if interval > 1:
                        interval_text = f" (every {interval} weeks)"
                    
                    formatted.append(f"Every {weekday}{time_info}, {date_range}{interval_text}")
                    
                    # Add note about exceptions if present
                    exceptions = rule.get('rcrExceptions', [])
                    if exceptions and isinstance(exceptions, list) and len(exceptions) > 0:
                        exception_dates = []
                        for exc in exceptions:
                            if isinstance(exc, dict) and exc.get('rcrExcDate'):
                                exc_date = exc.get('rcrExcDate')
                                # Format: MM/DD
                                exception_dates.append(f"{exc_date.split('-')[1]}/{exc_date.split('-')[2]}")
                        
                        if exception_dates:
                            formatted.append(f"Except: {', '.join(exception_dates)}")
        
        return formatted or ["Date not specified"]
    
    def clean_html(self, html_content):
        """Clean HTML content to plain text"""
        if not html_content:
            return ""
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            text = html.unescape(text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Truncate long descriptions
            if len(text) > 200:
                last_sentence = text[:200].rfind('.')
                if last_sentence > 50:
                    return text[:last_sentence+1]
                return text[:197] + "..."
            
            return text
        except Exception as e:
            self._log("ERROR", f"Error cleaning HTML: {str(e)}")
            return "Error processing content"
    
    def reduce_event(self, event):
        """Reduce a single event to essential information"""
        try:
            title = event.get('title', '')
            content = self.clean_html(event.get('content', ''))
            uri = event.get('uri', '')
            
            location_obj = event.get('location', {})
            location = location_obj.get('name', 'Location not specified') if location_obj else 'Location not specified'
            
            acf_event = event.get('acfGroupEvent', {})
            occasions = acf_event.get('occasions', [])
            recurring_rules = acf_event.get('rcrRules', [])
            
            # Get formatted dates, including recurring events
            dates = self.format_dates(occasions, recurring_rules)
            
            return {
                "title": title,
                "summary": content,
                "location": location,
                "dates": dates,
                "uri": uri
            }
        except Exception as e:
            self._log("ERROR", f"Error reducing event: {str(e)}")
            return {"title": event.get('title', 'Unknown event'), "error": "Processing error"}
    
    def process_events(self, events_data):
        """Process and reduce all events"""
        reduced_events = []
        
        for event in events_data:
            if isinstance(event, dict):
                reduced_event = self.reduce_event(event)
                reduced_events.append(reduced_event)
        
        return reduced_events
    
    def ask_gemini_about_events(self, query: str) -> str:
        """Ask Gemini about events based on query"""
        # Log the query
        self._log("USER", f"Events query: {query}")
        
        # Get current date
        current_date = time.strftime("%A, %Y-%m-%d")
        
        # Get and process events data
        events_data = self.get_events_data()
        if not events_data:
            return "Sorry, I couldn't retrieve any event data at this time."
        
        # Count tokens in original data
        original_events_json = json.dumps(events_data, ensure_ascii=False)
        original_tokens = self.count_tokens(original_events_json)
        
        # Reduce events to save tokens
        reduced_events = self.process_events(events_data)
        
        # Count tokens in reduced data
        reduced_events_json = json.dumps(reduced_events, ensure_ascii=False)
        reduced_tokens = self.count_tokens(reduced_events_json)
        
        # Calculate token reduction
        token_reduction = original_tokens - reduced_tokens
        token_reduction_percent = (token_reduction / original_tokens * 100) if original_tokens > 0 else 0
        
        # Log token counts
        token_stats = (
            f"Event tokens - Original: {original_tokens}, "
            f"Reduced: {reduced_tokens}, "
            f"Saved: {token_reduction} ({token_reduction_percent:.1f}%)"
        )
        self._log("TOKENS", token_stats)
        print(token_stats)
        
        # Simplified system prompt
        system_prompt = """Du är en expert på evenemang i Falkenbergs kommun. Besvara frågan om evenemang baserat på den data som tillhandahålls.

        Current date: {current_date}

Format för svar:
**Evenemang:**
- **[TITEL]**: [BESKRIVNING]. Datum: [DATUM]. Plats: [PLATS]. URI: [URI]
- **[TITEL]**: [BESKRIVNING]. Datum: [DATUM]. Plats: [PLATS]. URI: [URI]

Prioritera relevans och var koncis men informativ."""
        
        # Create context and prompt
        context = reduced_events_json
        full_prompt = f"{system_prompt}\n\nFråga: {query}\n\nEventdata: {context}"
        
        # Count tokens in prompt
        prompt_tokens = self.count_tokens(full_prompt)
        self._log("TOKENS", f"Event prompt tokens: {prompt_tokens}")
        print(f"Event prompt tokens: {prompt_tokens}")
        
        # Log that we're sending a request
        self._log("SYSTEM", "Sending event request to Gemini")
        
        try:
            # Generate response
            start_time = time.time()
            response = self.model.generate_content(full_prompt)
            end_time = time.time()
            
            # Log response time
            response_time = end_time - start_time
            self._log("SYSTEM", f"Gemini event response time: {response_time:.2f} seconds")
            print(f"Gemini event response time: {response_time:.2f} seconds")
            
            # Count tokens in response
            response_tokens = self.count_tokens(response.text)
            self._log("TOKENS", f"Event response tokens: {response_tokens}")
            print(f"Event response tokens: {response_tokens}")
            
            # Log the Gemini response (truncated)
            self._log("GEMINI", response.text[:100] + "..." if len(response.text) > 100 else response.text)
            
            return response.text
        except Exception as e:
            error_msg = f"Error from Gemini for events: {str(e)}"
            self._log("ERROR", error_msg)
            return f"Ett fel uppstod: {str(e)}"
    
    # PAGES FUNCTIONS
    
    def fetch_pages_data(self) -> List[Dict[str, Any]]:
        """Fetch pages data from CMS GraphQL API"""
        query = """
        query Pages {
          pages(first: 2000, where: { status: PUBLISH, language: SV }) {
            nodes {
              content
              date
              title
              uri
            }
          }
        }
        """
        
        try:
            response = requests.post(
                self.cms_url,
                json={"query": query},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                nodes = data.get("data", {}).get("pages", {}).get("nodes", [])
                return nodes
            else:
                self._log("ERROR", f"Failed to fetch pages: HTTP {response.status_code}")
                return []
        except Exception as e:
            self._log("ERROR", f"Error fetching pages: {str(e)}")
            return []
    
    def get_pages_data(self) -> List[Dict[str, Any]]:
        """Get pages data (from cache if valid)"""
        current_time = time.time()
        
        if (self.pages_cache["data"] is None or 
            current_time - self.pages_cache["last_updated"] > self.pages_cache["cache_duration"]):
            self.refresh_pages_data()
        
        return self.pages_cache["data"] or []
    
    def refresh_pages_data(self) -> None:
        """Refresh the pages data and update cache"""
        fresh_data = self.fetch_pages_data()
        self.pages_cache["data"] = fresh_data
        self.pages_cache["last_updated"] = time.time()
        self._log("SYSTEM", f"Refreshed pages data. Total pages: {len(fresh_data)}")
    
    def reduce_page(self, page):
        """Reduce a single page to essential information"""
        try:
            title = page.get('title', '')
            content = self.clean_html(page.get('content', ''))
            uri = page.get('uri', '')
            date = page.get('date', '')
            
            return {
                "title": title,
                "content": content,
                "uri": uri,
                "date": date
            }
        except Exception as e:
            self._log("ERROR", f"Error reducing page: {str(e)}")
            return {"title": page.get('title', 'Unknown page'), "error": "Processing error"}
    
    def process_pages(self, pages_data):
        """Process and reduce all pages"""
        reduced_pages = []
        
        for page in pages_data:
            if isinstance(page, dict):
                reduced_page = self.reduce_page(page)
                reduced_pages.append(reduced_page)
        
        return reduced_pages
    
    def ask_gemini_about_pages(self, query: str) -> str:
        """Ask Gemini about pages based on query"""
        # Log the query
        self._log("USER", f"Pages query: {query}")
        
        # Get and process pages data
        pages_data = self.get_pages_data()
        if not pages_data:
            return "Sorry, I couldn't retrieve any page data at this time."
        
        # Count tokens in original data
        original_pages_json = json.dumps(pages_data, ensure_ascii=False)
        original_tokens = self.count_tokens(original_pages_json)
        
        # Reduce pages to save tokens
        reduced_pages = self.process_pages(pages_data)
        
        # Count tokens in reduced data
        reduced_pages_json = json.dumps(reduced_pages, ensure_ascii=False)
        reduced_tokens = self.count_tokens(reduced_pages_json)
        
        # Calculate token reduction
        token_reduction = original_tokens - reduced_tokens
        token_reduction_percent = (token_reduction / original_tokens * 100) if original_tokens > 0 else 0
        
        # Log token counts
        token_stats = (
            f"Page tokens - Original: {original_tokens}, "
            f"Reduced: {reduced_tokens}, "
            f"Saved: {token_reduction} ({token_reduction_percent:.1f}%)"
        )
        self._log("TOKENS", token_stats)
        print(token_stats)
        
        # System prompt for pages
        system_prompt = """Du är en expert på Falkenbergs kommun och dess webbplats. Besvara frågan baserat på innehållet från webbsidorna på falkenberg.se. 

Fokusera på att ge ett detaljerat och korrekt svar baserat på informationen från webbsidorna. Ange uri/länk till relevanta sidor.

Prioritera relevans och var koncis men informativ."""
        
        # Create context and prompt
        context = reduced_pages_json
        full_prompt = f"{system_prompt}\n\nFråga: {query}\n\nWebbsidesdata: {context}"
        
        # Count tokens in prompt
        prompt_tokens = self.count_tokens(full_prompt)
        self._log("TOKENS", f"Pages prompt tokens: {prompt_tokens}")
        print(f"Pages prompt tokens: {prompt_tokens}")
        
        # Log that we're sending a request
        self._log("SYSTEM", "Sending pages request to Gemini")
        
        try:
            # Generate response
            start_time = time.time()
            response = self.model.generate_content(full_prompt)
            end_time = time.time()
            
            # Log response time
            response_time = end_time - start_time
            self._log("SYSTEM", f"Gemini pages response time: {response_time:.2f} seconds")
            print(f"Gemini pages response time: {response_time:.2f} seconds")
            
            # Count tokens in response
            response_tokens = self.count_tokens(response.text)
            self._log("TOKENS", f"Pages response tokens: {response_tokens}")
            print(f"Pages response tokens: {response_tokens}")
            
            # Log the Gemini response (truncated)
            self._log("GEMINI", response.text[:100] + "..." if len(response.text) > 100 else response.text)
            
            return response.text
        except Exception as e:
            error_msg = f"Error from Gemini for pages: {str(e)}"
            self._log("ERROR", error_msg)
            return f"Ett fel uppstod: {str(e)}"
    
    # COMMON FUNCTIONS
    
    def schedule_refresh(self, interval_hours: int = 3) -> None:
        """Schedule regular refreshes of both event and page data"""
        import threading
        
        def refresh_loop():
            while True:
                time.sleep(interval_hours * 3600)  # Convert hours to seconds
                self.refresh_events_data()
                self.refresh_pages_data()
        
        # Start the refresh loop in a background thread
        refresh_thread = threading.Thread(target=refresh_loop, daemon=True)
        refresh_thread.start()
        self._log("SYSTEM", f"Scheduled refresh every {interval_hours} hours")