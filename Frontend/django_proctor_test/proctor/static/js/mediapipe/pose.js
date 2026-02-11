export function calculateHeadPose(m) {
  const sy = Math.sqrt(m[0] * m[0] + m[4] * m[4]);
  const pitch = Math.atan2(m[9], m[10]);
  const yaw = Math.atan2(-m[8], sy);
  const roll = Math.atan2(m[4], m[0]);

  return {
    pitch: -pitch * 57.3,
    yaw: -yaw * 57.3,
    roll: roll * 57.3
  };
}

/**
 * Compute normalized bounding box from face landmarks.
 * @param {Array<{x:number, y:number}>} landmarks
 * @returns {{x:number, y:number, w:number, h:number}}
 */
export function getBoundingBox(landmarks) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;

  for (const p of landmarks) {
    minX = Math.min(minX, p.x);
    minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x);
    maxY = Math.max(maxY, p.y);
  }

  return {
    x: minX,
    y: minY,
    w: maxX - minX,
    h: maxY - minY
  };
}

