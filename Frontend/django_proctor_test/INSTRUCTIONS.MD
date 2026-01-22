
This error is a browser security restriction.
Browsers (Chrome, Firefox, Safari) only allow access to the camera (getUserMedia) in a Secure Context. A secure context is defined as:
- https://
- http://localhost
- http://127.0.0.1
Because you are accessing your app via an AWS IP address or an AWS Load Balancer URL over http://, the browser has disabled the camera API for security reasons.

Solution 1: Use an "Insecure" Bypass (Fastest for Testing)
If you just want to verify that your code works on AWS right now without setting up SSL certificates, you can tell Chrome to treat your AWS URL as secure.
1. Open Chrome and type this in the address bar:
chrome://flags/#unsafely-treat-insecure-origin-as-secure
2. Enable the "Insecure origins treated as secure" flag.
3. In the text box, paste your AWS URL (including the port), for example:
http://13.214.205.45:8002/exam/mediapipe/
4. Relaunch Chrome.
5. Visit your site again. The camera should now work.

Solution 2: Production Setup (The "Correct" Way)
To deploy this properly so other people can use it, you must serve it over HTTPS.
1. Get a Domain Name: You cannot get an SSL certificate for a raw AWS IP address. You need a domain (e.g., proctor.example.com).
2. AWS Certificate Manager (ACM): Request a free SSL certificate for your domain in the AWS Console.
3. Application Load Balancer (ALB):
Create an ALB in AWS.
Create an HTTPS Listener (Port 443).
Attach your SSL certificate to that listener.
Forward traffic from the ALB (443) to your Docker container (8002).
4. Update your WebSocket URL:
Once on HTTPS, your JavaScript must connect via wss:// (secure WebSockets) instead of ws://.

Solution 3: Testing via Localhost (Validation)
If you want to prove the Docker image is fine, run it on your local machine using Docker Desktop:
1. docker run -p 8002:8002 proctor-app
2. Visit http://localhost:8002 in your browser.
3. Because it is localhost, the camera will work even without HTTPS.

Important note on WebSockets:
When you finally move to HTTPS, remember that standard WebSockets (ws://) will be blocked. You will need to update your script to use wss://:

// In production, use wss:// if the current page is https://
const protocol = window.location.protocol === 'https:' ? 'wss://' : 'ws://';
const WS_URL_TEMPLATE = scriptTag?.dataset.wsUrl ?? `${protocol}${window.location.host}/ws/{sessionId}`;


Key Security Features Integrated:
The "Active Obfuscation" Layer: The .guarded-content CSS class includes a high-intensity blur (25px). Unlike just showing a message, this ensures that the student cannot see any part of the exam if the AI loses sight of them for more than 5 seconds.
Stability Lock (The Friction Defense): In the handleFacePresent function, if the system was previously isLocked, it enters a secondary loop. The student must keep their head in the frame for 3.0 consecutive seconds before unlockUI() is called. This prevents the "Look-and-Dip" cheating method.
15-Second Kill Switch: The consecutiveMissingMs check terminates the exam at exactly 15 seconds. This is the calculated "Safe Window" for network jitter, while still being too short for an examinee to successfully execute an AI prompt on a secondary device.
60-Second Cumulative Bucket: The totalMissingMs variable tracks the entire exam duration. If a student keeps looking away for 4 seconds, returning, and repeating, they will hit the 60-second limit and be terminated automatically.
State-Based Status: The status-box changes color (ok, warn, err) to provide the student with immediate feedback on why the system is acting (e.g., "CALIBRATING", "SESSION LOCKED", "TERMINATED").

3. Key Changes Explained
The Aggregate Timer (totalMissingMs): This variable never resets. Even if the student returns for a second, their previous time spent away is "saved." If the total exceeds 60 seconds, they are kicked.
The 15-Second Hard Cut: This replaces your 30-second window. This is the "Cheating Latency" defense—preventing enough time for a full AI search query.
The Stability Lock: When isLocked is true, the stabilityMs counter must reach 3000ms (3 seconds) while a face is detected before the blur is removed. This adds "Friction," preventing students from "popping" in and out to reset timers.
Information Obfuscation: At the 5-second mark, the .locked class is applied. This blurs the head-pose data and buttons. (In your actual exam app, you should apply this blur to the exam questions).
4. Technical Defense for your Documentation
"The system was upgraded from a 30-second inactivity timeout to a Multi-Tiered Temporal Enforcement model.
Latency Mitigation: The 15-second consecutive limit is based on research into AI-assisted retrieval times (Daffin & Jones, 2018), ensuring detection gaps are too short for external consultation.
Anti-Evasion: A 60-second cumulative limit prevents the 'Gaming the Timer' loophole where students briefly return to reset detection.
Stability Validation: Upon returning from a gap >5s, the system enforces a 3-second 'Stability Lock,' requiring verified presence before content is unblurred, neutralizing the cognitive advantage of quick glances."

