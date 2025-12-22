// Get configuration from the script tag's data attributes
const scriptTag = document.getElementById('proctor-script');
const MODEL_PATH = scriptTag?.dataset.modelPath ?? '';
const WASM_PATH = scriptTag?.dataset.wasmPath ?? '';
const WS_URL_TEMPLATE = scriptTag?.dataset.wsUrl ?? 'ws://127.0.0.1:8001/ws/{sessionId}';

let faceLandmarker;
let ws = null;
let sessionId = null;
const video = document.getElementById("video");
const statusPre = document.getElementById("status");

/** Initialize the Metadata-First Proctoring Engine **/
async function initProctoring() {
    // Prefer globals if provided by script tag; otherwise dynamically import the tasks package.
    let FilesetResolverRef = window.FilesetResolver;
    let FaceLandmarkerRef = window.FaceLandmarker;

    if (!FilesetResolverRef || !FaceLandmarkerRef) {
        try {
            const mod = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3');
            FilesetResolverRef = mod.FilesetResolver;
            FaceLandmarkerRef = mod.FaceLandmarker;
        } catch (err) {
            console.error('Failed to load MediaPipe Tasks from CDN:', err);
            throw err;
        }
    }

    const vision = await FilesetResolverRef.forVisionTasks(WASM_PATH || 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm');
    faceLandmarker = await FaceLandmarkerRef.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_PATH, delegate: "GPU" },
        numFaces: 1,
        minFaceDetectionConfidence: 0.4,
        minTrackingConfidence: 0.4,
        outputFacialTransformationMatrixes: true,
        runningMode: "VIDEO" // Optimized for 5-8 FPS tracking
    });
}

/** The Core Processing Loop **/
async function processFrames() {
    if (video.paused || video.ended || !faceLandmarker) return;

    const results = faceLandmarker.detectForVideo(video, performance.now());

    if (results?.facialTransformationMatrixes?.length > 0) {
        const matrix = Array.from(results.facialTransformationMatrixes[0].data || []);
        
        // 1. Send lightweight JSON Metadata (Yaw, Pitch, Roll data)
        if (ws?.readyState === WebSocket.OPEN) {
            try { ws.send(JSON.stringify({ type: "metadata", matrix: matrix })); }
            catch (e) { console.warn('WS send metadata failed', e); }
        }

        // 2. Conditional Binary Logic: Only send image if head turns significantly
        const zVal = matrix[2] ?? 0;
        if (Math.abs(zVal) > 0.5) { // Simple threshold check
            sendBinaryEvidence();
        }
        statusPre.innerText = "Monitoring: Active (Metadata Stream)";
    } else {
        statusPre.innerText = "Monitoring: No face tracked";
    }
    
    if (sessionId) window.requestAnimationFrame(processFrames);
}

/** Capture and send evidence only when flagged **/
function sendBinaryEvidence() {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
        if (blob && ws?.readyState === WebSocket.OPEN) {
            try { ws.send(blob); }
            catch (e) { console.warn('WS send blob failed', e); }
        }
    }, "image/jpeg", 0.5); // Compressed for speed
}

document.getElementById("startBtn").onclick = async () => {
    try {
        await initProctoring();
    } catch (err) {
        statusPre.innerText = 'Initialization error: see console';
        return;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    sessionId = crypto.randomUUID();

    const wsUrl = WS_URL_TEMPLATE.replace('{sessionId}', sessionId);
    ws = new WebSocket(wsUrl);

    ws.onopen = () => { statusPre.innerText = `WS connected to ${wsUrl}`; };
    ws.onmessage = (evt) => {
        // optional: display server messages
        try {
            const payload = JSON.parse(evt.data);
            console.log('Server message:', payload);
        } catch (e) {
            console.log('Server raw message:', evt.data);
        }
    };
    ws.onclose = () => { statusPre.innerText = 'WS closed'; };
    ws.onerror = (e) => { statusPre.innerText = 'WS error'; console.error(e); };

    document.getElementById("endBtn").disabled = false;
    video.onloadeddata = () => processFrames();
};

document.getElementById("endBtn").onclick = () => {
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    ws = null;
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(t => t.stop());
        video.srcObject = null;
    }
    sessionId = null;
    statusPre.innerText = "Disconnected";
};
