import requests
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

FASTAPI_BASE = "http://127.0.0.1:8001"


@csrf_exempt
def start_session(request):
    """Start a new proctoring session."""
    res = requests.post(f"{FASTAPI_BASE}/sessions/start")
    return JsonResponse(res.json())

"""
request.FILES.get("frame") → this assumes the browser is submitting a file object via a <form> with enctype="multipart/form-data".
The Django view then posts it to FastAPI using:
files={"image_file": frame}

Even if your camera is live (from a webcam), how you send the frame to the backend determines the endpoint used:

/frame expects file uploads (UploadFile)
→ Your Django view is using request.FILES, so it sends the frame as a file.
/frame_base64 expects Base64 strings (Form)
→ You would need to convert the webcam image to a Base64 string in JavaScript and send it as a POST form field b64.

Why it works for a webcam
Even if you capture a frame from the camera, if your frontend JS or HTML form converts it to a file object (like a Blob) and submits via <input type="file"> or FormData.append('frame', blob), then FastAPI will use /frame.
You don’t need /frame_base64 unless:
-> You are capturing frames in the browser via canvas.toDataURL() (which gives Base64)
-> And you want to send that string directly without converting to a Blob
"""
@csrf_exempt
def send_frame(request):
    """Send image frame to FastAPI."""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"})

    session_id = request.POST.get("session_id")
    frame = request.FILES.get("frame")

    if not session_id:
        return JsonResponse({"error": "Missing session_id"})

    if not frame:
        return JsonResponse({"error": "Missing frame image file"})

    res = requests.post(
        f"{FASTAPI_BASE}/sessions/{session_id}/frame",
        files={"image_file": frame}
    )

    return JsonResponse(res.json())


@csrf_exempt
def send_frame_base64(request):
    """Send Base64 image to FastAPI using /sessions/{id}/frame_base64"""
    if request.method != "POST":
        return JsonResponse({"error": "POST required"})

    session_id = request.POST.get("session_id")  
    b64 = request.POST.get("b64")

    if not session_id:
        return JsonResponse({"error": "Missing session_id"})

    if not b64:
        return JsonResponse({"error": "Missing base64 frame"})

    res = requests.post(
        f"{FASTAPI_BASE}/sessions/{session_id}/frame_base64",
        files={"b64": (None, b64)}   # IMPORTANT FIX
    )

    try:
        return JsonResponse(res.json())
    except Exception:
        return JsonResponse({
            "error": "FastAPI did not return JSON",
            "status": res.status_code,
            "text": res.text
        })

@csrf_exempt
def end_session(request):
    """End session and save logs."""
    session_id = request.POST.get("session_id")

    if not session_id:
        return JsonResponse({"error": "Missing session_id"})

    res = requests.post(
        f"{FASTAPI_BASE}/sessions/{session_id}/end"
    )

    return JsonResponse(res.json())


def get_report(request):
    """Fetch final session report."""
    session_id = request.GET.get("session_id")

    if not session_id:
        return JsonResponse({"error": "Missing session_id"})

    res = requests.get(
        f"{FASTAPI_BASE}/sessions/{session_id}/report"
    )

    return JsonResponse(res.json())

from django.shortcuts import render

def test_page(request):
    # return render(request, "proctor/exam_test.html")
    return render(request, "proctor/exam_test_v2.html")

def ws_test_page(request):
    return render(request, "proctor/ws_test.html")
