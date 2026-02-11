import { SAFE_ZONE } from "../config.js";

/**
 * Draw face bounding box and calibration guide overlay.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {HTMLCanvasElement} canvas
 * @param {Object|null} box - Face bounding box {x,y,w,h}
 * @param {boolean} isViolation - Whether a rule violation is active
 * @param {boolean} isCalibrating - Whether calibration mode is active
 */
export function drawBoundingBox(ctx, canvas, box, isViolation, isCalibrating) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw calibration safe zone even if face is missing
  if (isCalibrating) {
    ctx.strokeStyle = "#00FFFF";
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 5]);

    ctx.strokeRect(
      canvas.width * SAFE_ZONE.xMin,
      canvas.height * SAFE_ZONE.yMin,
      canvas.width * (SAFE_ZONE.xMax - SAFE_ZONE.xMin),
      canvas.height * (SAFE_ZONE.yMax - SAFE_ZONE.yMin)
    );

    ctx.setLineDash([]);
  }

  if (!box) return;

  ctx.lineWidth = 3;
  ctx.strokeStyle = isViolation ? "red" : "#00FF00";
  ctx.strokeRect(box.x, box.y, box.w, box.h);
}
