/**
 * Detect head pose and framing violations.
 * PURE LOGIC ONLY — no DOM access.
 */

import { ANGLE_LIMITS, SAFE_ZONE } from "../config.js";

/**
 * @param {{yaw:number, pitch:number, roll:number}} pose
 * @param {{x:number, y:number, w:number, h:number}|null} box
 * @param {{width:number, height:number}} frame
 * @returns {string[]} List of violation messages
 */
export function detectViolations(pose, box, frame) {
  const violations = [];

  // --- Head pose violations ---
  if (pose.pitch > ANGLE_LIMITS.PITCH_DOWN) violations.push("Looking Down");
  if (pose.pitch < ANGLE_LIMITS.PITCH_UP) violations.push("Looking Up");
  if (pose.yaw > ANGLE_LIMITS.YAW_LEFT) violations.push("Looking Left");
  if (pose.yaw < ANGLE_LIMITS.YAW_RIGHT) violations.push("Looking Right");
  if (pose.roll > ANGLE_LIMITS.ROLL_LEFT) violations.push("Head tilted Left");
  if (pose.roll < ANGLE_LIMITS.ROLL_RIGHT) violations.push("Head tilted Right");

  // --- Framing violations ---
  if (box && frame?.width && frame?.height) {
    const faceCX = (box.x + box.w / 2) / frame.width;
    const faceCY = (box.y + box.h / 2) / frame.height;

    if (faceCX < SAFE_ZONE.xMin) violations.push("Too far Right");
    if (faceCX > SAFE_ZONE.xMax) violations.push("Too far Left");
    if (faceCY < SAFE_ZONE.yMin) violations.push("Too High");
    if (faceCY > SAFE_ZONE.yMax) violations.push("Too Low");
  }

  return violations;
}
