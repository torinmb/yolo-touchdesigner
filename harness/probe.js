// Page-side probe (injected before the app): socket and WebGPU activity per frame.
// Never modifies payloads. Read back with window.__probe.
(() => {
  const P = (window.__probe = { recv: [], send: [], map: [], submits: [], writes: [], copies: 0, t0: performance.timeOrigin });
  const now = () => performance.timeOrigin + performance.now();
  // WebSocket receive (onmessage setter) and send
  const desc = Object.getOwnPropertyDescriptor(WebSocket.prototype, 'onmessage');
  Object.defineProperty(WebSocket.prototype, 'onmessage', {
    set(fn) { desc.set.call(this, fn && ((ev) => { const d = ev.data; P.recv.push([now(), typeof d === 'string' ? d.length : d.byteLength]); return fn.call(this, ev); })); },
    get() { return desc.get.call(this); },
  });
  const origSend = WebSocket.prototype.send;
  WebSocket.prototype.send = function (d) {
    const text = typeof d === 'string';
    P.send.push([now(), text ? d.length : (d.byteLength ?? d.size ?? 0), text ? (d.match(/"type":"(\w+)"/) || [])[1] || 'text' : 'binary', this.bufferedAmount]);
    return origSend.call(this, d);
  };
  if (window.GPUBuffer) {
    const origMap = GPUBuffer.prototype.mapAsync;
    GPUBuffer.prototype.mapAsync = function (...a) {
      const t = now(), size = this.size;
      return origMap.apply(this, a).then((r) => { P.map.push([t, now() - t, size]); return r; });
    };
  }
  if (window.GPUQueue) {
    const s = GPUQueue.prototype.submit; GPUQueue.prototype.submit = function (...a) { P.submits.push(now()); return s.apply(this, a); };
    const w = GPUQueue.prototype.writeBuffer; GPUQueue.prototype.writeBuffer = function (b, o, d, ...r) { P.writes.push([now(), d.byteLength ?? d.length]); return w.call(this, b, o, d, ...r); };
  }
  if (window.GPUCommandEncoder) {
    const c = GPUCommandEncoder.prototype.copyBufferToBuffer; GPUCommandEncoder.prototype.copyBufferToBuffer = function (...a) { P.copies++; return c.apply(this, a); };
  }
})();
