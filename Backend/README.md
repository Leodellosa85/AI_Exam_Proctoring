uvicorn app.main:app --reload --port 8001

What are yaw and pitch?
They are angles describing the orientation of the head in 3D space:
Angle	                What it measures	                Example / Intuition
Yaw	                    Left-right rotation (turning head)	Shaking your head “no” → yaw changes
Pitch	                Up-down rotation (nodding)	        Nodding “yes” → pitch changes
Roll (not used here)	Tilt of the head to the shoulder	Tilting head sideways → roll changes

yaw, pitch = detector.analyze_head_pose(img)

detector.analyze_head_pose(img) returns:
yaw → how much the person is turning their head left/right
pitch → how much the person is looking up/down

These values are usually in degrees or radians, depending on the library.

How it’s used for lookaway detection
handle_lookaway(session, yaw, pitch, LOOKAWAY_MIN_ANGLE)

LOOKAWAY_MIN_ANGLE is a threshold.
If abs(yaw) or abs(pitch) exceeds the threshold → the system flags the user as looking away from the screen.
Example:
yaw = 50° → the user turned head far to the side → might be cheating
pitch = -30° → the user looks down → maybe reading notes

So yaw and pitch are just standard terms from 3D geometry and aviation to describe rotations in space. They’re used in computer vision to quantify where a person is looking.


PIL means Python Imaging Library.
-> It is the standard image-processing library in Python, used for:
- opening images (JPEG, PNG, WEBP, etc.)
- resizing images
- converting formats
- cropping
- drawing
- converting images into tensors for AI models

PIL itself is old and discontinued, so today we use its modern replacement:
✅ Pillow
-> Pillow is a maintained, modern fork of PIL.
-> But everyone still casually refers to it as “PIL” because the import name is:
from PIL import Image

-> So you still write PIL, but the actual package installed is Pillow.



What Is a WebSocket?
-> A WebSocket is a special type of network connection that stays open and allows real-time two-way communication between:
🔄 Browser (frontend) ↔ FastAPI backend

Unlike normal HTTP:
Feature	            HTTP	                    WebSocket
Connection	        Opens → Sends → Closes	    Opens once and stays open
Direction	        One-way	                    Two-way (send + receive anytime)
Best used for	    Requests, forms, APIs	    Live video, real-time messaging, streaming
🎯 Why your proctoring system needs WebSockets
-> Sending 30 frames per second using normal HTTP = slow, laggy, expensive
WebSockets allow:
- continuous frames
- faster detection
- real-time cheating analysis
- immediate feedback (looking away, multiple faces, etc.)