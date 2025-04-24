import asyncio
import aiohttp
from bs4 import BeautifulSoup
import googlemaps
from datetime import datetime
import python_weather
from googlesearch import search as Google_Search_sync

# Utility function for weather
async def get_weather(location: str, api_key=None, socketio=None, client_sid=None) -> dict | None:
    """ Fetches current weather and emits update via SocketIO. """
    if not location or len(location.strip()) == 0:
        print("Error: Empty location provided to get_weather")
        return {"error": "No location provided. Please specify a city or location."}
        
    print(f"🌡️ Getting weather for location: '{location}'")
    async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
        try:
            # Handle multi-part location strings
            weather = await client.get(location)
            weather_data = {
                'location': location,
                'current_temp_f': weather.temperature,
                'precipitation': weather.precipitation,
                'description': weather.description,
            }
            print(f"Weather data fetched: {weather_data}")

            # --- Emit weather_update if socketio is provided ---
            if socketio and client_sid:
                print(f"--- Emitting weather_update event for SID: {client_sid} ---")
                socketio.emit('weather_update', weather_data, room=client_sid)
            # --- End Emit ---

            return weather_data # Return data for Gemini

        except Exception as e:
            error_message = f"Error fetching weather for '{location}': {e}"
            print(error_message)
            if socketio and client_sid:
                socketio.emit('error', {'message': error_message}, room=client_sid)
            return {
                "error": f"Could not fetch weather for '{location}'.",
                "error_details": str(e),
                "suggestion": "Please try with a more specific location format, such as 'City, State, Country'"
            } # Return detailed error info

# Utility function for travel duration (sync version)
def _sync_get_travel_duration(origin: str, destination: str, mode: str = "driving", maps_api_key=None) -> str:
    if not maps_api_key or maps_api_key == "YOUR_PROVIDED_KEY":
        print("Error: Google Maps API Key is missing or invalid.")
        return "Error: Missing or invalid Google Maps API Key configuration."
    try:
        gmaps = googlemaps.Client(key=maps_api_key)
        now = datetime.now()
        print(f"Requesting directions: From='{origin}', To='{destination}', Mode='{mode}'")
        directions_result = gmaps.directions(origin, destination, mode=mode, departure_time=now)
        if directions_result:
            leg = directions_result[0]['legs'][0]
            duration_text = "Not available"
            if mode == "driving" and 'duration_in_traffic' in leg:
                duration_text = leg['duration_in_traffic']['text']
                result = f"Estimated travel duration ({mode}, with current traffic): {duration_text}"
            elif 'duration' in leg:
                duration_text = leg['duration']['text']
                result = f"Estimated travel duration ({mode}): {duration_text}"
            else:
                result = f"Duration information not found in response for {mode}."
            print(f"Directions Result: {result}")
            return result
        else:
            print(f"No route found from {origin} to {destination} via {mode}.")
            return f"Could not find a route from {origin} to {destination} via {mode}."
    except Exception as e:
        print(f"An unexpected error occurred during travel duration lookup: {e}")
        return f"An unexpected error occurred: {e}"

# Utility function for travel duration (async wrapper)
async def get_travel_duration(origin: str, destination: str, mode: str = "driving", maps_api_key=None, socketio=None, client_sid=None) -> dict:
    """ Async wrapper to get travel duration and emit map update via SocketIO. """
    print(f"Received request for travel duration from: {origin} to: {destination}, Mode: {mode}")
    if not mode:
        mode = "driving"

    try:
        result_string = await asyncio.to_thread(
            _sync_get_travel_duration, origin, destination, mode, maps_api_key
        )

        # --- Emit map_update from here ---
        if socketio and client_sid and not result_string.startswith("Error"): # Only emit if successful
            map_payload = {
                'destination': destination,
                'origin': origin
            }
            print(f"--- Emitting map_update event for SID: {client_sid} ---")
            socketio.emit('map_update', map_payload, room=client_sid)
        # --- End Emit ---

        return {"duration_result": result_string} # Return result for Gemini

    except Exception as e:
        print(f"Error calling _sync_get_travel_duration via to_thread: {e}")
        return {"duration_result": f"Failed to execute travel duration request: {e}"}

# Utility function for fetching and extracting webpage content
async def _fetch_and_extract_snippet(session, url: str) -> dict | None:
    """
    Fetches HTML from a URL, extracts title, meta description,
    and concatenates text from paragraph tags.
    Returns a dictionary or None on failure.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    title = "No Title Found"
    snippet = "No Description Found"
    page_text_summary = "Could not extract page text." # Default value

    try:
        async with session.get(url, headers=headers, timeout=15, ssl=False) as response: # Increased timeout slightly
            if response.status == 200:
                html_content = await response.text()
                soup = BeautifulSoup(html_content, 'lxml')

                # --- Extract Title ---
                title_tag = soup.find('title')
                if title_tag and title_tag.string:
                    title = title_tag.string.strip()

                # --- Extract Meta Description ---
                description_tag = soup.find('meta', attrs={'name': 'description'})
                if description_tag and description_tag.get('content'):
                    snippet = description_tag['content'].strip()

                # --- Extract Text from Paragraphs ---
                try:
                    paragraphs = soup.find_all('p') # Find all <p> tags
                    # Join the text content of all paragraphs, stripping whitespace
                    full_page_text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

                    # Basic Processing/Summarization
                    max_len = 1500 # Limit the amount of text returned
                    if len(full_page_text) > max_len:
                        page_text_summary = full_page_text[:max_len] + "..."
                    else:
                        page_text_summary = full_page_text

                    if not page_text_summary: # Handle case where no paragraph text was found
                        page_text_summary = "No paragraph text found on page."

                except Exception as text_ex:
                    print(f"  Error extracting paragraph text from {url}: {text_ex}")
                    # Keep default "Could not extract..." message

                print(f"  Extracted: Title='{title}', Snippet='{snippet[:50]}...', Text='{page_text_summary[:50]}...' from {url}")
                # --- Return enriched dictionary ---
                return {
                    "url": url,
                    "title": title,
                    "meta_snippet": snippet, # Renamed for clarity
                    "page_content_summary": page_text_summary # Added page text
                }
            else:
                print(f"  Failed to fetch {url}: Status {response.status}")
                return None # Return None on non-200 status

    # --- Error handling ---
    except asyncio.TimeoutError:
        print(f"  Timeout fetching {url}")
    except aiohttp.ClientError as e:
        print(f"  ClientError fetching {url}: {e}")
    except Exception as e:
        print(f"  Error processing {url}: {e}")

    # Return None if any exception occurred before successful extraction
    return None

# Utility function for Google search (sync version)
def _sync_Google_Search(query: str, num_results: int = 5) -> list:
    print(f"Performing synchronous Google search for: '{query}'")
    try:
        results = list(Google_Search_sync(term=query, num_results=num_results, lang="en", timeout=1))
        print(f"Found {len(results)} results.")
        return results
    except Exception as e:
        print(f"Error during Google search for '{query}': {e}")
        return []

# Utility function for Google search (async wrapper)
async def get_search_results(query: str, socketio=None, client_sid=None) -> dict:
    """
    Async wrapper for Google search. Fetches URLs, then retrieves
    title, meta snippet, and a summary of page paragraph text for each.
    Emits results via SocketIO.
    Returns a dictionary containing a list of result objects.
    """
    print(f"Received request for Google search with page content fetch: '{query}'")
    fetched_results = [] # This will store dicts: {"url":..., "title":..., "meta_snippet":..., "page_content_summary":...}
    try:
        # Step 1: Get URLs
        search_urls = await asyncio.to_thread(
            _sync_Google_Search, query, num_results=5
        )
        if not search_urls:
            print("No URLs found by Google Search.")
            # --- EMIT EMPTY RESULTS TO FRONTEND ---
            if socketio and client_sid:
                print(f"--- Emitting empty search_results_update event for SID: {client_sid} ---")
                socketio.emit('search_results_update', {"results": [], "query": query}, room=client_sid)
            # --- END EMIT ---
            return {"results": []} # Return for Gemini

        # Step 2: Fetch content concurrently
        print(f"Fetching content for {len(search_urls)} URLs...")
        async with aiohttp.ClientSession() as session:
            tasks = [_fetch_and_extract_snippet(session, url) for url in search_urls]
            results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)

        # Step 3: Process results (filter Nones and Exceptions)
        for result in results_from_gather:
            if isinstance(result, dict): # Successfully fetched data
                fetched_results.append(result)
            elif isinstance(result, Exception):
                print(f"   An error occurred during content fetching task: {result}")
            # else: result is None (fetch/parse failed, already logged in helper)

        print(f"Finished fetching content. Got {len(fetched_results)} results.")

        # --- EMIT RESULTS TO FRONTEND ---
        if socketio and client_sid:
             print(f"--- Emitting search_results_update event with {len(fetched_results)} results for SID: {client_sid} ---")
             # Send the query along with the results for context
             emit_payload = {"query": query, "results": fetched_results}
             socketio.emit('search_results_update', emit_payload, room=client_sid)
        # --- END EMIT ---

    except Exception as e:
        print(f"Error running get_search_results for '{query}': {e}")
        # Optionally emit an error event to the frontend here as well
        if socketio and client_sid:
             socketio.emit('search_results_error', {"query": query, "error": str(e)}, room=client_sid)
        return {"error": f"Failed to execute Google search with page content: {str(e)}"} # Return for Gemini

    # Format the final result for Gemini
    response_payload = {
        "results": fetched_results
    }
    print(f"Custom Google search function for '{query}' returning {len(fetched_results)} processed results to Gemini.")
    return response_payload 