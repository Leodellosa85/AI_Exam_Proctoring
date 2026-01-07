
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
http://3.21.123.45:8002
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
