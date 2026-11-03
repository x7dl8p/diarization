// AudioWorklet: hands each 128-frame block (mixed to mono) to the page, which forwards it to the server.
class Capture extends AudioWorkletProcessor {
  process(inputs) {
    const chans = inputs[0];
    if (chans && chans.length) {
      const n = chans[0].length;
      const mono = new Float32Array(n);
      for (const c of chans) for (let i = 0; i < n; i++) mono[i] += c[i] / chans.length;
      this.port.postMessage(mono, [mono.buffer]);
    }
    return true;
  }
}
registerProcessor("capture", Capture);
