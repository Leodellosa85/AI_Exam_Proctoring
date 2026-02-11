/**
 * LivenessClient
 * ----------------
 * Handles WebSocket communication with the liveness backend.
 * Responsible only for transport & protocol framing.
 */
export class LivenessClient {
  constructor(url) {
    this.ws = new WebSocket(url);
    this.messageHandler = null;

    this.ws.onmessage = e => {
      if (this.messageHandler) {
        this.messageHandler(JSON.parse(e.data));
      }
    };
  }

  /**
   * Register handler for liveness results.
   * @param {(data: object) => void} cb
   */
  onMessage(cb) {
    this.messageHandler = cb;
  }

  /**
   * Send face crops to backend.
   */
  sendCrops(mini, facebag) {
    if (this.ws.readyState !== WebSocket.OPEN) return;

    this.ws.send(JSON.stringify({
      type: "liveness_payload",
      mini: true,
      facebag: true
    }));

    this.ws.send(mini.blobV2);
    this.ws.send(facebag);
  }

  close() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.close();
    }
  }
}
