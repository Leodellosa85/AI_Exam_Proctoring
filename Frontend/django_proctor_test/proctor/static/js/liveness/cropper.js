import {
  MINI_INPUT_SIZE,
  FACEBAG_INPUT_SIZE,
  MAX_SCALE_V2,
  MAX_SCALE_V1SE,
  MAX_SCALE_FACEBAG,
  MIN_SCALE,
  MIN_FACE_SIZE
} from "../config.js";

/**
 * Adapt crop scale based on detected face width.
 */
export function adaptiveScale(boxW, maxScale) {
  if (boxW > 220) return maxScale;
  if (boxW > 160) return Math.min(maxScale, 2.0);
  if (boxW > 120) return Math.min(maxScale, 1.6);
  return MIN_SCALE;
}

/**
 * Crop and resize face region from video frame.
 */
export function cropFace(video, box, targetSize, scale) {
  const vw = video.videoWidth;
  const vh = video.videoHeight;

  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;

  const cropSize = Math.max(box.w, box.h) * scale;
  if (!isFinite(cropSize) || cropSize <= 0) {
    return Promise.resolve(null);
  }

  let x = cx - cropSize / 2;
  let y = cy - cropSize / 2;

  x = Math.max(0, Math.min(x, vw - cropSize));
  y = Math.max(0, Math.min(y, vh - cropSize));

  const canvas = document.createElement("canvas");
  canvas.width = targetSize;
  canvas.height = targetSize;

  const ctx = canvas.getContext("2d", { alpha: false });
  ctx.imageSmoothingEnabled = false;

  ctx.drawImage(
    video,
    x, y, cropSize, cropSize,
    0, 0, targetSize, targetSize
  );

  return new Promise(res => canvas.toBlob(res, "image/png"));
}

/**
 * Generate dual crops for MiniFASNet V2 and V1SE.
 */
export async function generateMiniCrops(video, box) {
  if (!box || box.w < MIN_FACE_SIZE || box.h < MIN_FACE_SIZE) return null;

  const scaleV2 = adaptiveScale(box.w, MAX_SCALE_V2);

  const [blobV2] = await Promise.all([
    cropFace(video, box, MINI_INPUT_SIZE, scaleV2)
  ]);

  if (!blobV2) return null;
  return { blobV2 };
}

/**
 * Generate crop for FaceBagNet.
 */
export async function generateFaceBagnetCrop(video, box) {
  if (!box || box.w < MIN_FACE_SIZE || box.h < MIN_FACE_SIZE) return null;

  const scale = adaptiveScale(box.w, MAX_SCALE_FACEBAG);
  return cropFace(video, box, FACEBAG_INPUT_SIZE, scale);
}
