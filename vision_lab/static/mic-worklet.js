// Downsamples the mic to 16 kHz float32 and posts ~32 ms chunks to the page.
class Mic16k extends AudioWorkletProcessor {
  constructor() { super(); this.ratio = sampleRate / 16000; this.pos = 0; this.out = []; }
  process(inputs) {
    const ch = inputs[0][0];
    if (!ch) return true;
    for (; this.pos < ch.length; this.pos += this.ratio) this.out.push(ch[Math.floor(this.pos)]);
    this.pos -= ch.length;
    if (this.out.length >= 512) { this.port.postMessage(new Float32Array(this.out)); this.out = []; }
    return true;
  }
}
registerProcessor('mic-16k', Mic16k);
