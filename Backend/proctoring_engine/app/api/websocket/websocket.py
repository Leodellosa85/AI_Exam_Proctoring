from fastapi import WebSocket, WebSocketDisconnect
from ...core.session_manager import get_session, create_session
from ...core.spoof_engine import SpoofEngine
import json

class LivenessWebSocket:
    def __init__(self):
        self.engine = SpoofEngine()

    async def handle(self, websocket: WebSocket, session_id: str):
        await websocket.accept()

        session = get_session(session_id)
        if not session:
            session_id, session = create_session()

        last_header = None

        try:
            while True:
                message = await websocket.receive()

                if message["type"] == "websocket.disconnect":
                    print(f"[WS] Client disconnected: {session_id}")
                    break

                # ---- TEXT FRAME (JSON HEADER) ----
                if "text" in message:
                    header = json.loads(message["text"])

                    if header.get("type") != "liveness_payload":
                        continue

                    last_header = header

                # ---- BINARY FRAME (IMAGE BYTES) ----
                elif "bytes" in message:
                    if not last_header:
                        continue  # ignore stray binary

                    blobs = []

                    if last_header.get("mini"):
                        blobs.append(message["bytes"])
                        blobs.append((await websocket.receive())["bytes"])

                    if last_header.get("facebag"):
                        blobs.append((await websocket.receive())["bytes"])

                    # ---- Run spoof engine ----
                    result = self.engine.process(
                        mini_v2=blobs[0] if len(blobs) > 0 else None,
                        facebag=blobs[-1] if len(blobs) > 1 else None,
                        session=session,
                        session_id=session_id
                    )

                    await websocket.send_json({
                        "type": "spoof_result",
                        **result
                    })

                    last_header = None

        except WebSocketDisconnect:
            print(f"[WS] disconnected: {session_id}")
