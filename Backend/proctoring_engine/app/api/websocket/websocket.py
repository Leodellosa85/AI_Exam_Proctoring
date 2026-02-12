import json
from fastapi import WebSocket, WebSocketDisconnect
from ...core.session_manager import get_session, create_session
from ...core.spoof_engine import SpoofEngine

class LivenessWebSocket:
    def __init__(self):
        # The engine should be initialized once to keep models in memory
        self.engine = SpoofEngine()

    async def handle(self, websocket: WebSocket, session_id: str):
        await websocket.accept()

        session = get_session(session_id)
        if not session:
            # Fallback if session manager doesn't have it yet
            session_id, session = create_session()

        # State management for multi-frame payloads
        blobs = []
        expected_blobs = 0
        last_header = None

        try:
            while True:
                # Receive the generic message
                message = await websocket.receive()

                # Handle Disconnection
                if message.get("type") == "websocket.disconnect":
                    print(f"[WS] Client disconnected: {session_id}")
                    break

                # ---- 1. TEXT FRAME (The JSON Protocol Header) ----
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "liveness_payload":
                            last_header = data
                            # Reset accumulator for new payload
                            blobs = []
                            # According to your frontend: 
                            # Mini (1 blob: blobV2) + FaceBag (1 blob) = 2 binary frames
                            expected_blobs = 2 
                            continue
                    except json.JSONDecodeError:
                        print(f"[WS] Invalid JSON received in session {session_id}")
                        continue

                # ---- 2. BINARY FRAME (The Image Data) ----
                elif "bytes" in message:
                    if expected_blobs > 0:
                        blobs.append(message["bytes"])

                        # Check if we have received all parts of the current payload
                        if len(blobs) == expected_blobs:
                            # Mapping: 
                            # blobs[0] is the mini crop (V2)
                            # blobs[1] is the facebag crop
                            
                            result = self.engine.process(
                                mini_v2=blobs[0],
                                facebag=blobs[1],
                                session=session,
                                session_id=session_id
                            )

                            # Send result back to frontend
                            await websocket.send_json({
                                "type": "spoof_result",
                                **result
                            })

                            # Reset state for next header
                            blobs = []
                            expected_blobs = 0
                            last_header = None
                    else:
                        print(f"[WS] Received unexpected binary data in session {session_id}")

        except WebSocketDisconnect:
            print(f"[WS] Disconnected: {session_id}")
        except Exception as e:
            print(f"[WS] Error in websocket handler: {str(e)}")
            # Optional: send error back to client