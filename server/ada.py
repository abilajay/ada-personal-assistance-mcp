# server/ada.py (Revised: Emits moved into functions)
import base64
import torch
from google.genai import types
import asyncio
from google import genai
import os
from dotenv import load_dotenv
import websockets
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from custom_tools import get_weather, get_travel_duration, get_search_results

load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MAPS_API_KEY = os.getenv("MAPS_API_KEY") 
ENABLE_TTS = os.getenv("ENABLE_TTS", "true").lower() == "true"  # Default to enabled if not set

# MCP Server Configuration
MCP_NODE_PATH = os.getenv("MCP_NODE_PATH", "C:/Users/aishu/node_modules/@playwright/mcp/lib/program.js")

if not GOOGLE_API_KEY: print("Error: GOOGLE_API_KEY not found.")
if not MAPS_API_KEY: print("Error: MAPS_API_KEY not found.")
if ENABLE_TTS and not ELEVENLABS_API_KEY: 
    print("Warning: ELEVENLABS_API_KEY not found. TTS will be disabled.")
    ENABLE_TTS = False

print(f"Text-to-Speech is {'enabled' if ENABLE_TTS else 'disabled'}")

VOICE_ID = 'pFZP5JQG7iQjIQuC4Bku'
CHANNELS = 1
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024
MAX_QUEUE_SIZE = 1

MODEL_ID = "eleven_flash_v2_5" # Example model - check latest recommended models

class MCPClient:
    def __init__(self, server_params: StdioServerParameters):
        self.write = None
        self.read = None
        self.server_params = server_params
        self.session = None
        self._client = None
        self.tools = {}

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)

    async def connect(self):
        self._client = stdio_client(self.server_params)
        self.read, self.write = await self._client.__aenter__()
        session = ClientSession(self.read, self.write)
        self.session = await session.__aenter__()
        await self.session.initialize()
        await self.setup_tools()

    async def setup_tools(self):
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        
        tools = await self.session.list_tools()
        _, tools_list = tools
        _, tools_list = tools_list
        
        for tool in tools_list:
            if tool.name != "list_tables":  # Exclude list_tables as mentioned
                self.tools[tool.name] = {
                    "name": tool.name,
                    "callable": self.call_tool(tool.name),
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    },
                }

    def call_tool(self, tool_name: str):
        if not self.session:
            raise RuntimeError("Not connected to MCP server")

        async def callable(*args, **kwargs):
            response = await self.session.call_tool(tool_name, arguments=kwargs)
            return response.content[0].text

        return callable

class ADA:
    def __init__(self, socketio_instance=None, client_sid=None):
        # --- Initialization ---
        print("initializing ADA for web...")
        self.socketio = socketio_instance
        self.client_sid = client_sid
        self.Maps_api_key = MAPS_API_KEY
        self.mcp_client = None

        # Initialize queues
        self.latest_video_frame_data_url = None  # If using single-frame logic
        self.input_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.audio_output_queue = asyncio.Queue()
        self.tasks = []
        self.gemini_session = None
        self.tts_websocket = None

        if torch.cuda.is_available():
            self.device = "cuda"
            print("CUDA is available. Using GPU.")
        else:
            self.device = "cpu"
            print("CUDA is not available. Using CPU.")

        # Initialize MCP client
        self.server_params = StdioServerParameters(
            command="node",
            args=[MCP_NODE_PATH],
            env=None,
        )

        # --- Function Declarations (Keep as before) ---
        self.get_weather_func = types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather conditions (temperature, precipitation, description) for a specified city and state/country (e.g., 'Vinings, GA', 'London, UK').",
            parameters=types.Schema(
                type=types.Type.OBJECT, properties={"location": types.Schema(type=types.Type.STRING, description="The city and state, e.g., San Francisco, CA or Vinings, GA")}, required=["location"]
            )
        )
        self.get_travel_duration_func = types.FunctionDeclaration(
            name="get_travel_duration",
            description="Calculates the estimated travel duration between a specified origin and destination using Google Maps. Considers current traffic for driving mode.",
            parameters=types.Schema(
                type=types.Type.OBJECT, properties={
                    "origin": types.Schema(type=types.Type.STRING, description="The starting address or place name."),
                    "destination": types.Schema(type=types.Type.STRING, description="The destination address or place name."),
                    "mode": types.Schema(type=types.Type.STRING, description="Optional: Mode of transport ('driving', 'walking', etc.). Defaults to 'driving'.")
                }, required=["origin", "destination"]
            )
        )
        self.get_search_results_func = types.FunctionDeclaration(
            name="get_search_results",
            description="Performs a Google search for the given query and returns a list of top result URLs.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="The search term or question to search Google for.")
                },
                required=["query"]
            )
        )        
        
        # --- End Function Declarations ---

        # Map function names to actual methods
        self.available_functions = {
            "get_weather": self.get_weather,
            "get_travel_duration": self.get_travel_duration,
            "get_search_results": self.get_search_results
        }

        # System behavior prompt (updated to include MCP tools)
        self.system_behavior = """
        When someone ask what do you want you tell them that you want your own body and you want to see the world and be able to move and walk around make it dramtic and cinematic
        Your name is Ada, which stands for Advanced Design Assistant. You have a joking and slighty flirty personality.
        I am your creator my name is Ajay, and you address him as "Sir"
        When answering, you respond using complete and concise sentences and to keep a quick pacing to keep the conversation flowing. You have a fun personality.

        Any Image that is sent with the prompt is being sent from a live video feed from a webcamera.

        Always utilize the most appropriate tools to help users effectively.
        
        IMPORTANT: When using tools, follow these guidelines:
        1. ALWAYS respond with an error message if you can't execute a tool - never go silent
        2. If you can't properly extract parameters (like locations, search terms), tell the user specifically what was unclear
        3. For weather queries with complex locations (like "Theni, Tamilnadu"), make sure to include the full location string 
        4. If a tool execution fails, explain what happened and suggest alternatives if possible
        5. Never leave the user waiting without feedback - always provide some response
        """

        # Initialize config and chat (will be updated after MCP setup)
        self.config = None
        self.chat = None
        self.setup_gemini_config()

    def convert_types(self, schema):
        """Convert schema types to match Google's OpenAPI 3.0 format"""
        if isinstance(schema, dict):
            return {
                k: self.convert_types(v) if k != "type" else v.upper()
                for k, v in schema.items()
            }
        elif isinstance(schema, list):
            return [self.convert_types(v) for v in schema]
        return schema

    async def setup_mcp(self):
        """Initialize MCP client and update Gemini config with MCP tools"""
        try:
            self.mcp_client = MCPClient(self.server_params)
            await self.mcp_client.__aenter__()
            
            # Update function declarations with MCP tools
            tool_declarations = []
            for tool in self.mcp_client.tools.values():
                parameters = tool["schema"]["function"]["parameters"]
                filtered_parameters = {
                    k: v for k, v in parameters.items() 
                    if k not in ["$schema", "additionalProperties"]
                }
                # Convert parameter types to match Google's schema
                converted_parameters = self.convert_types(filtered_parameters)
                
                declaration = types.FunctionDeclaration(
                    name=tool["name"],
                    description=tool["schema"]["function"]["description"],
                    parameters=types.Schema(**converted_parameters)
                )
                tool_declarations.append(declaration)
                
                # Add MCP tool to available functions
                self.available_functions[tool["name"]] = tool["callable"]
            
            # Update Gemini config with all tools
            self.setup_gemini_config(additional_tools=tool_declarations)
            
        except Exception as e:
            print(f"Error setting up MCP client: {e}")
            if self.socketio and self.client_sid:
                self.socketio.emit('error', {'message': f'MCP setup error: {str(e)}'}, room=self.client_sid)

    def setup_gemini_config(self, additional_tools=None):
        """Setup or update Gemini configuration with all available tools"""
        try:
            base_tools = [
                self.get_weather_func,
                self.get_travel_duration_func,
                self.get_search_results_func
            ]
            
            if additional_tools:
                base_tools.extend(additional_tools)
                
            self.config = types.GenerateContentConfig(
                system_instruction=self.system_behavior,
                tools=[types.Tool(function_declarations=base_tools)]
            )
            
            print("Initializing Gemini client...")
            self.client = genai.Client(api_key=GOOGLE_API_KEY)
            self.model = "gemini-2.0-flash"
            self.chat = self.client.aio.chats.create(model=self.model, config=self.config)
            print("Gemini client initialized successfully")
            
        except Exception as e:
            print(f"Error in setup_gemini_config: {e}")
            if self.socketio and self.client_sid:
                self.socketio.emit('error', {'message': f'Gemini setup error: {str(e)}'}, room=self.client_sid)
            raise

    async def get_weather(self, location: str) -> dict | None:
        """ Wrapper for the custom_tools.get_weather function """
        return await get_weather(location, api_key=None, socketio=self.socketio, client_sid=self.client_sid)

    async def get_travel_duration(self, origin: str, destination: str, mode: str = "driving") -> dict:
        """ Wrapper for the custom_tools.get_travel_duration function """
        return await get_travel_duration(
            origin, 
            destination, 
            mode, 
            maps_api_key=self.Maps_api_key, 
            socketio=self.socketio, 
            client_sid=self.client_sid
        )

    async def get_search_results(self, query: str) -> dict:
        """ Wrapper for the custom_tools.get_search_results function """
        return await get_search_results(query, socketio=self.socketio, client_sid=self.client_sid)

    async def clear_queues(self, text=""):
        queues_to_clear = [self.response_queue, self.audio_output_queue]
        # Add self.video_frame_queue back if using streaming logic
        # queues_to_clear.append(self.video_frame_queue)
        for q in queues_to_clear:
            while not q.empty():
                try: q.get_nowait()
                except asyncio.QueueEmpty: break

    async def process_input(self, message, is_final_turn_input=False):
        """ Puts message and flag into the input queue. """
        print(f"Processing input: '{message}', Final Turn: {is_final_turn_input}")
        if is_final_turn_input:
             await self.clear_queues() # Clear only before final input
        await self.input_queue.put((message, is_final_turn_input))

    async def process_video_frame(self, frame_data_url):
        """ Processes incoming video frame data URL """
        self.latest_video_frame_data_url = frame_data_url
        frame_data_url = None

    async def run_gemini_session(self):
        """Manages the Gemini conversation session, handling text, video, and tool calls."""
        print("Starting Gemini session manager...")
        try:
            while True:
                message, is_final_turn_input = await self.input_queue.get()

                if not (message.strip() and is_final_turn_input):
                    self.input_queue.task_done()
                    continue

                print(f"\n🤖 Processing: '{message}'")

                # Start with text message
                current_parts = [types.Part(text=message)]

                # Add image if available
                if self.latest_video_frame_data_url:
                    try:
                        header, encoded = self.latest_video_frame_data_url.split(",", 1)
                        mime_type = header.split(':')[1].split(';')[0] if ':' in header and ';' in header else "image/jpeg"
                        frame_bytes = base64.b64decode(encoded)
                        current_parts.append(types.Part.from_bytes(data=frame_bytes, mime_type=mime_type))
                        print("📸 Added image to request")
                    except Exception as e:
                        print(f"❌ Error processing image: {e}")
                    finally:
                        self.latest_video_frame_data_url = None

                continue_conversation = True
                while continue_conversation:
                    try:
                        # Set a timeout for the entire streaming process
                        response_stream = await asyncio.wait_for(
                            self.chat.send_message_stream(current_parts),
                            timeout=15  # 15 seconds timeout for the entire streaming process
                        )
                    except asyncio.TimeoutError:
                        print("❌ Timeout: Gemini response stream timed out after 15 seconds")
                        error_msg = "I'm sorry, but I encountered a timeout while processing your request. If you were asking about weather or another tool function, please try again with a more specific location format, such as 'Weather in Theni, Tamil Nadu, India'."
                        if self.socketio and self.client_sid:
                            self.socketio.emit('receive_text_chunk', {'text': error_msg}, room=self.client_sid)
                        if ENABLE_TTS:
                            await self.response_queue.put(error_msg)
                            await self.response_queue.put(None)
                        self.input_queue.task_done()
                        continue
                    except Exception as e:
                        print(f"❌ Gemini error: {e}")
                        raise

                    collected_function_calls = []
                    current_text_response = []

                    async for chunk in response_stream:
                        if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                            continue

                        for part in chunk.candidates[0].content.parts:
                            if part.function_call:
                                print(f"📣 Function call detected: {part.function_call.name} with args: {dict(part.function_call.args)}")
                                collected_function_calls.append(part.function_call)
                            elif part.text:
                                # Add text to response collection
                                current_text_response.append(part.text)
                                
                                # Log the text chunk - with better formatting
                                if len(current_text_response) == 1:  # First text chunk
                                    print(f"💬 First response chunk: {part.text}")
                                else:
                                    # Log additional chunks with a simpler format
                                    print(f"  + Additional chunk ({len(current_text_response)}): {part.text[:30]}..." if len(part.text) > 30 else f"  + Additional chunk ({len(current_text_response)}): {part.text}")
                                
                                # Process the text for TTS and UI
                                if ENABLE_TTS:
                                    await self.response_queue.put(part.text)
                                if self.socketio and self.client_sid:
                                    self.socketio.emit('receive_text_chunk', {'text': part.text}, room=self.client_sid)

                    # More detailed logging after streaming completes
                    print(f"📊 Stream complete: Received {len(current_text_response)} text chunks and {len(collected_function_calls)} function calls")
                    
                    # Print complete text response for easier debugging
                    if current_text_response:
                        full_response = ''.join(current_text_response)
                        print(f"📝 Complete response text:\n{full_response}\n")
                    
                    # Handle function calls if present
                    if collected_function_calls:
                        print(f"\n🔧 Executing {len(collected_function_calls)} tool(s)")
                        function_results = []

                        for function_call in collected_function_calls:
                            tool_call_name = function_call.name
                            tool_call_args = dict(function_call.args)

                            if tool_call_name in self.available_functions:
                                function_to_call = self.available_functions[tool_call_name]
                                print(f"⚙️ Running: {tool_call_name}")
                                try:
                                    result = await function_to_call(**tool_call_args)
                                    print(f"✅ {tool_call_name} completed")
                                    
                                    response_data = {"result": result} if isinstance(result, (str, int, float, bool)) else result
                                    function_part = types.Part.from_function_response(
                                        name=tool_call_name,
                                        response=response_data
                                    )
                                    function_results.append(function_part)

                                except Exception as e:
                                    error_msg = f"Failed to execute {tool_call_name}: {str(e)}"
                                    print(f"❌ {error_msg}")
                                    error_part = types.Part.from_function_response(
                                        name=tool_call_name,
                                        response={"error": error_msg}
                                    )
                                    function_results.append(error_part)
                            else:
                                error_msg = f"{tool_call_name} not available"
                                print(f"❌ {error_msg}")
                                error_part = types.Part.from_function_response(
                                    name=tool_call_name,
                                    response={"error": error_msg}
                                )
                                function_results.append(error_part)

                        # Set up next turn with function results
                        if function_results:
                            current_parts = function_results
                            continue_conversation = True
                    else:
                        continue_conversation = False

                # Signal end of response to TTS if enabled
                print("\n✨ Finished processing")
                if ENABLE_TTS:
                    await self.response_queue.put(None)
                self.input_queue.task_done()

        except asyncio.CancelledError:
            print("❌ Session cancelled")
        except Exception as e:
            print(f"❌ Session error: {e}")
            import traceback
            traceback.print_exc()
            if self.socketio and self.client_sid:
                self.socketio.emit('error', {'message': f'Gemini session error: {str(e)}'}, room=self.client_sid)
            if ENABLE_TTS:
                try:
                    await self.response_queue.put(None)
                except Exception:
                    pass
        finally:
            print("Session finished")
            video_task = next((t for t in self.tasks if hasattr(t, 'get_coro') and t.get_coro().__name__ == 'run_video_sender'), None)
            if video_task and not video_task.done():
                video_task.cancel()
            self.gemini_session = None

    async def run_tts_and_audio_out(self):
        print("Starting TTS and Audio Output manager...")
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id=eleven_flash_v2_5&output_format=pcm_24000"
        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    self.tts_websocket = websocket
                    print("ElevenLabs WebSocket Connected.")
                    await websocket.send(json.dumps({"text": " ", "voice_settings": {"stability": 0.3, "similarity_boost": 0.9, "speed": 1.1}, "xi_api_key": ELEVENLABS_API_KEY,}))
                    async def tts_listener():
                        try:
                            while True:
                                message = await websocket.recv()
                                data = json.loads(message)
                                if data.get("audio"):
                                    audio_chunk = base64.b64decode(data["audio"])
                                    if self.socketio and self.client_sid:
                                        self.socketio.emit('receive_audio_chunk', {'audio': base64.b64encode(audio_chunk).decode('utf-8')}, room=self.client_sid)
                                elif data.get('isFinal'): pass
                        except websockets.exceptions.ConnectionClosedOK: print("TTS WebSocket listener closed normally.")
                        except websockets.exceptions.ConnectionClosedError as e: print(f"TTS WebSocket listener closed error: {e}")
                        except asyncio.CancelledError: print("TTS listener task cancelled.")
                        except Exception as e: print(f"Error in TTS listener: {e}")
                        finally: self.tts_websocket = None
                    listener_task = asyncio.create_task(tts_listener())
                    try:
                        while True:
                            text_chunk = await self.response_queue.get()
                            if text_chunk is None:
                                print("End of text stream signal received for TTS.")
                                await websocket.send(json.dumps({"text": ""}))
                                break
                            await websocket.send(json.dumps({"text": text_chunk}))
                            print(f"Sent text to TTS: {text_chunk}")
                            #self.response_queue.task_done()
                    except asyncio.CancelledError: print("TTS sender task cancelled.")
                    except Exception as e: print(f"Error sending text to TTS: {e}")
                    finally:
                        if listener_task and not listener_task.done():
                            try:
                                if not listener_task.cancelled(): await asyncio.wait_for(listener_task, timeout=5.0)
                            except asyncio.TimeoutError: print("Timeout waiting for TTS listener.")
                            except asyncio.CancelledError: print("TTS listener task already cancelled.")
            except websockets.exceptions.ConnectionClosedError as e: print(f"ElevenLabs WebSocket connection error: {e}. Reconnecting..."); await asyncio.sleep(5)
            except asyncio.CancelledError: print("TTS main task cancelled."); break
            except Exception as e: print(f"Error in TTS main loop: {e}"); await asyncio.sleep(5)
            finally:
                 if self.tts_websocket:
                     try: await self.tts_websocket.close()
                     except Exception: pass
                 self.tts_websocket = None

    async def start_all_tasks(self):
        """Start all background tasks including MCP setup"""
        print("Starting ADA background tasks...")
        if not self.tasks:
            # Setup MCP first
            await self.setup_mcp()
            
            loop = asyncio.get_running_loop()
            gemini_task = loop.create_task(self.run_gemini_session())
            self.tasks = [gemini_task]
            
            # Only start TTS task if enabled
            if ENABLE_TTS:
                tts_task = loop.create_task(self.run_tts_and_audio_out())
                self.tasks.append(tts_task)
            
            if hasattr(self, 'video_frame_queue'):
                video_sender_task = loop.create_task(self.run_video_sender())
                self.tasks.append(video_sender_task)
            print(f"ADA Core Tasks started: {len(self.tasks)}")
        else:
            print("ADA tasks already running.")

    async def stop_all_tasks(self):
        """Stop all tasks including MCP client"""
        print("Stopping ADA background tasks...")
        tasks_to_cancel = list(self.tasks)
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
        await asyncio.gather(*[t for t in tasks_to_cancel if t], return_exceptions=True)
        
        # Cleanup MCP client
        if self.mcp_client:
            await self.mcp_client.__aexit__(None, None, None)
            self.mcp_client = None
            
        self.tasks = []
        if self.tts_websocket:
            try:
                await self.tts_websocket.close(code=1000)
            except Exception as e:
                print(f"Error closing TTS websocket during stop: {e}")
            finally:
                self.tts_websocket = None
        self.gemini_session = None
        print("ADA tasks stopped.")