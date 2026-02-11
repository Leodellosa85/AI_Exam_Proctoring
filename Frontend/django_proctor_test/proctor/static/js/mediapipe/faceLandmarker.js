import {
  FaceLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

/**
 * Initialize MediaPipe FaceLandmarker.
 * @param {string} modelPath - Path to face_landmarker.task
 * @param {string} wasmPath - Path to MediaPipe wasm directory
 * @returns {Promise<FaceLandmarker>}
 */
export async function initFaceLandmarker(modelPath, wasmPath) {
  const vision = await FilesetResolver.forVisionTasks(wasmPath);

  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: modelPath,
      delegate: "CPU"
    },
    numFaces: 1,
    minFaceDetectionConfidence: 0.6,
    minTrackingConfidence: 0.5,
    outputFacialTransformationMatrixes: true,
    runningMode: "VIDEO"
  });
}

/**
 * Run face detection on a video frame.
 * @param {FaceLandmarker} faceLandmarker
 * @param {HTMLVideoElement} video
 * @param {number} timestamp - performance.now()
 */
export function detectFace(faceLandmarker, video, timestamp) {
  return faceLandmarker.detectForVideo(video, timestamp);
}
