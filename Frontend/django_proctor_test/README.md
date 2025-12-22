python manage.py runserver 8000

INSTALLED_APPS: Check for daphne and channels.
ASGI_APPLICATION: Pointing to your asgi.py.
CHANNEL_LAYERS: Configured for Redis.
MEDIA_ROOT: Configured to save the snapshots.

Since you transitioned from a standard Django website to a Real-Time WebSocket App that saves Images, you need to configure settings.py to handle three specific things:
Asynchronous Support (ASGI/Channels)
Redis Communication
Image File Storage

Configure Redis (Channel Layers)
Django Channels needs a "background worker" to pass messages. We use Redis for this.
Add this block anywhere in settings.py:

Requirement: You must have Redis running (docker run -p 6379:6379 redis) and install the package:


Phase 1: Input Acquisition (Client-Side)
- Step: The application accesses the webcam via the browser's navigator.mediaDevices API.
- Constraint: Resolution is locked to 640x480.
- Why? Processing High-Definition (1080p) video in JavaScript is too slow for real-time analysis. 640x480 is the optimal balance between accuracy and speed (low latency).

Phase 2: AI Inference (Edge Computing)
- Step: The video frame is passed to Google MediaPipe Face Landmarker.
- Technology: WebAssembly (WASM) running on the CPU.
- Process: The model scans the frame and returns:
    - Face Landmarks: 478 x/y coordinates of the face mesh.
    - Transformation Matrix: Mathematical data representing head rotation in 3D space.

Phase 3: Data Processing & Calibration
- Step: Converting raw AI data into usable metrics.
- Mathematics: The system uses trigonometric functions (Math.atan2) to convert the Transformation Matrix into Euler Angles:
    - Pitch: Looking Up / Down.
    - Yaw: Turning Left / Right.
    - Roll: Tilting Head.
- Calibration (The "Zeroing" Effect):
    - Context: Not every student sits perfectly straight, and webcams are often tilted.
    - Action: During startup, the system records the user's natural position (basePose).
    - Runtime: Every subsequent frame is calculated as CurrentAngle - BaseAngle. This ensures fairness regardless of the laptop's hinge angle.

Phase 4: The Rule Engine (Violation Detection)
- Step: Determining if the student is misbehaving.
- Logic: The system checks two types of violations:
    - Angle Violations: Is the head turned beyond the allowed threshold (e.g., > 30°)? Logic is split into Left/Right and Up/Down to handle sensor asymmetry.
    - Position Violations: Is the face center outside the "Safe Zone" (the central 60% of the screen)?
- Throttling: The AI runs at 8 FPS, but we only record violations once per second to prevent flooding the database with duplicate errors.

Phase 5: Evidence Capture
- Step: Creating proof of the violation.
- Action: If a violation triggers:
    - A hidden <canvas> draws the current video frame.
    - The image is converted to a Base64 JPEG string (compressed quality 0.5).
    - This string is bundled into a JSON object with the violation type and timestamp.

Phase 6: Transmission (WebSocket)
    - Step: Sending data to the backend.
    - Protocol: WebSocket (ws://) instead of HTTP.
    - Why? HTTP requires opening a new connection for every request (slow). WebSockets keep a permanent "pipe" open, allowing instant data transfer with near-zero overhead.

Phase 7: Server-Side Handling (Django)
    - Step: Receiving and Saving.
    - Component: Daphne (ASGI Server) receives the packet and passes it to consumers.py.
    Action:
        -Authentication: Verifies which student sent the data using the SessionID.
        - Decoding: Converts the Base64 string back into a binary image file.
        - Storage:
        - Saves the Image to the /media/ folder.
        - Saves the Log (Timestamp, Violation Type, Angles) to the ViolationLog table in SQLite/PostgreSQL.
