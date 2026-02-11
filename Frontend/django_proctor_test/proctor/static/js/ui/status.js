/**
 * Update main status banner text and style.
 * @param {HTMLElement} el - Status element
 * @param {string} text - Status message
 * @param {"ok"|"warn"|"err"} type - Status type
 */
export function setStatus(el, text, type = "ok") {
  el.textContent = text;
  el.className = `status ${type}`;
}

/**
 * Update yaw / pitch / roll display values.
 * @param {{yaw:number, pitch:number, roll:number}} pose
 * @param {{yaw:HTMLElement, pitch:HTMLElement, roll:HTMLElement}} els
 */
export function updateDisplayMetrics(pose, els) {
  els.yaw.textContent = `${pose.yaw.toFixed(1)}°`;
  els.pitch.textContent = `${pose.pitch.toFixed(1)}°`;
  els.roll.textContent = `${pose.roll.toFixed(1)}°`;
}

/**
 * Toggle liveness button state.
 * @param {object} state - Session state
 * @param {HTMLElement} btn - Toggle button
 */
export function toggleLiveness(state, btn) {
  state.useBackendLiveness = !state.useBackendLiveness;
  btn.textContent = `Liveness: ${state.useBackendLiveness ? "ON" : "OFF"}`;
  btn.style.background = state.useBackendLiveness ? "#22c55e" : "#6b7280";
}

/**
 * Update cumulative lost-time UI.
 * @param {number} totalMs - Total missing time in milliseconds
 * @param {HTMLElement} totalLostEl - Element displaying lost time
 */
export function updateTotalLostUI(totalMs, totalLostEl) {
  totalLostEl.textContent = `${Math.floor(totalMs / 1000)}s`;
}
