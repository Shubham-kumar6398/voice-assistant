/**
 * AudioWorklet processor: computes cross-correlation between mic (input 0)
 * and TTS reference (input 1) for echo detection.
 * Posts {corr, micRms, refRms} to the main thread every 128 frames.
 */
class CorrelatorProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._bufLen = 256;
    this._mic = new Float32Array(this._bufLen);
    this._ref = new Float32Array(this._bufLen);
    this._pos = 0;
  }

  process(inputs, outputs) {
    const mic = inputs[0] && inputs[0][0];
    const ref = inputs[1] && inputs[1][0];

    // Pass mic audio to output so the ScriptProcessor can read it
    if (outputs[0] && outputs[0][0] && mic) {
      outputs[0][0].set(mic);
    }

    const len = mic ? mic.length : 0;
    for (let i = 0; i < len; i++) {
      const idx = this._pos % this._bufLen;
      this._mic[idx] = mic ? mic[i] : 0;
      this._ref[idx] = ref ? ref[i] : 0;
      this._pos++;
    }

    // Report every 128 samples
    if (len > 0 && this._pos % 128 === 0) {
      let micSq = 0, refSq = 0, cross = 0;
      for (let i = 0; i < this._bufLen; i++) {
        micSq  += this._mic[i] * this._mic[i];
        refSq  += this._ref[i] * this._ref[i];
        cross  += this._mic[i] * this._ref[i];
      }
      const micRms = Math.sqrt(micSq / this._bufLen);
      const refRms = Math.sqrt(refSq / this._bufLen);
      const denom  = micRms * refRms;
      const corr   = denom > 1e-8 ? Math.abs(cross / (this._bufLen * denom)) : 0;
      this.port.postMessage({ corr, micRms, refRms });
    }

    return true;
  }
}

registerProcessor('correlator', CorrelatorProcessor);
