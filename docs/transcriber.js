/**
 * whisper-webCLI Transcription Engine
 *
 * Real, in-browser speech-to-text powered by Transformers.js (Whisper ONNX).
 * Everything runs on-device: model weights are downloaded once from the Hugging
 * Face Hub, cached by the browser, and inference happens locally via WebAssembly
 * (or WebGPU when available). No audio ever leaves the device.
 *
 * This module is intentionally decoupled from the UI so it can be reused
 * elsewhere (e.g. the ffmpeg-webCLI "Auto-Caption" feature). It is loaded as an
 * ES module inside a Web Worker.
 */

import {
  pipeline,
  env,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2';

// Always fetch models from the Hugging Face Hub and cache them in the browser.
env.allowLocalModels = false;
env.useBrowserCache = true;

// Map the UI model ids to concrete Whisper ONNX repositories on the HF Hub.
const MODEL_MAP = {
  tiny: 'Xenova/whisper-tiny',
  base: 'Xenova/whisper-base',
  small: 'Xenova/whisper-small',
  medium: 'Xenova/whisper-medium',
};

export class Transcriber {
  constructor() {
    this.pipe = null;
    this.loadedModel = null;
  }

  /**
   * Lazily load (and cache) the ASR pipeline for the requested model.
   * @param {string} modelId - UI model id (tiny/base/small/medium)
   * @param {(p: object) => void} [onProgress] - model download progress
   */
  async _ensureModel(modelId, onProgress) {
    const repo = MODEL_MAP[modelId] || MODEL_MAP.tiny;
    if (this.pipe && this.loadedModel === repo) {
      return;
    }

    // Apply ORT config to avoid nested worker SecurityError on CDN
    try {
      if (env.backends?.onnx?.wasm) {
        env.backends.onnx.wasm.numThreads = 1;
        env.backends.onnx.wasm.proxy = false;
      }
    } catch (_) {
      // Ignore if backends not yet initialized
    }

    // Dispose previous pipeline if switching models.
    if (this.pipe && typeof this.pipe.dispose === 'function') {
      try {
        await this.pipe.dispose();
      } catch (_) {
        /* ignore */
      }
    }

    this.pipe = await pipeline('automatic-speech-recognition', repo, {
      // Quantized weights keep the download small and inference fast.
      device: 'wasm',
      dtype: {
        encoder_model: 'q8',
        decoder_model_merged: 'q8',
      },
      progress_callback: (p) => {
        if (onProgress) onProgress(p);
      },
    });
    this.loadedModel = repo;
  }

  /**
   * Transcribe audio.
   * @param {Float32Array} audio - mono PCM samples
   * @param {object} options
   * @param {number} [options.sampleRate=16000] - sample rate of `audio`
   * @param {string} [options.model='tiny']
   * @param {string} [options.language='auto'] - ISO code or 'auto'
   * @param {boolean} [options.translate=false] - translate to English
   * @param {(p: object) => void} [options.onModelProgress]
   * @param {(p: object) => void} [options.onStatus]
   * @param {(p: object) => void} [options.onProgress] - streaming progress: { pct, label, segments }
   * @returns {Promise<Array<{start:number,end:number,text:string}>>}
   */
  async transcribe(audio, options = {}) {
    const {
      sampleRate = 16000,
      model = 'base',
      language = 'ru',
      translate = false,
      resumeOffset = 0,
      initialSegments = [],
      onModelProgress,
      onStatus,
      onProgress,
    } = options;

    if (onStatus) onStatus({ stage: 'loading-model', model });
    await this._ensureModel(model, onModelProgress);

    // Whisper expects 16 kHz mono float32.
    const audio16k = this._resampleTo16k(audio, sampleRate);

    if (onStatus) onStatus({ stage: 'transcribing' });

    // Stream results by processing audio in 30-second chunks with overlap.
    // This lets the UI show partial results as transcription progresses.
    const chunkDuration = 15; // seconds
    const overlapDuration = 0; // seconds overlap for context
    const sampleRate16k = 16000;
    const chunkSamples = chunkDuration * sampleRate16k;
    const overlapSamples = overlapDuration * sampleRate16k;
    const stride = chunkSamples - overlapSamples; // slide by 25 seconds

    const chunks = [];
    let offset = Math.min(audio16k.length, Math.max(0, Math.round(resumeOffset * 16000)));
    let globalTimeOffset = offset / 16000;
    const allSegments = initialSegments.slice();

    while (offset < audio16k.length) {
      const end = Math.min(offset + chunkSamples, audio16k.length);
      const chunkAudio = audio16k.slice(offset, end);

      let energy = 0, peak = 0;
      for (const x of chunkAudio) { if (!Number.isFinite(x)) throw new Error('Invalid PCM'); energy += x*x; peak = Math.max(peak, Math.abs(x)); }
      const rms = Math.sqrt(energy / chunkAudio.length);
      const result = peak < 0.00001 ? {text: '[Почти нулевой сигнал — проверить аудио]'} : await this.pipe(chunkAudio, {
        chunk_length_s: 0,
        stride_length_s: 0,
        return_timestamps: true,
        language: 'russian',
        task: 'transcribe',
        do_sample: false,
        max_new_tokens: 192,
        repetition_penalty: 1.1,
      });

      // Convert this chunk's results and adjust timestamps.
      const chunkSegments = this._toSegments(result, chunkAudio.length / sampleRate16k);
      for (const seg of chunkSegments) {
        // Shift timestamps by the global time offset.
        if (Number.isFinite(seg.start)) seg.start += globalTimeOffset;
        if (Number.isFinite(seg.end)) seg.end += globalTimeOffset;
        const words = seg.text.toLowerCase().match(/[\p{L}\p{N}]+/gu) || [];
        let repeated = false;
        for (let n=1;n<=8;n++) for(let j=0;j+4*n<=words.length;j++) {
          const phrase=words.slice(j,j+n).join(' ');
          if ([1,2,3].every(k=>words.slice(j+k*n,j+(k+1)*n).join(' ')===phrase)) repeated=true;
        }
        if (repeated) seg.text = '[Возможное зацикливание — сверить с оригиналом] ' + seg.text;
        allSegments.push(seg);
      }

      // Move to next chunk (with overlap context).
      globalTimeOffset += stride / sampleRate16k;
      offset += stride;

      // Emit progress: percentage and accumulated results.
      const pct = Math.round((offset / audio16k.length) * 100);
      if (onProgress) {
        onProgress({
          pct: Math.min(pct, 99), // Don't show 100 until fully done
          label: `Transcribing… ${pct}%`,
          segments: allSegments,
          completedSeconds: Math.min(offset, audio16k.length) / 16000,
        });
      }
    }

    // Final result with all segments.
    return allSegments;
  }

  /**
   * Convert a Transformers.js ASR result into timestamped segments.
   */
  _toSegments(result, totalDuration) {
    if (result && Array.isArray(result.chunks) && result.chunks.length) {
      // Pre-filter non-empty chunks and calculate duration for fallback
      const nonEmptyChunks = result.chunks.filter((chunk) => (chunk.text || '').trim().length > 0);
      
      
      return nonEmptyChunks
        .map((chunk, i) => {
          const [start, end] = chunk.timestamp || [];
          const text = (chunk.text || '').trim();
          
          // If timestamps are available, use them
          if (typeof start === 'number' && typeof end === 'number') {
            return {
              start,
              end,
              text,
            };
          }
          
          // Fallback: distribute segments evenly across total duration
          return {
            start: null,
            end: null,
            text,
          };
        });
    }

    // Fallback: single segment spanning the whole clip.
    const text = result && result.text ? result.text.trim() : '';
    return text ? [{ start: null, end: null, text }] : [];
  }

  /**
   * Resample mono Float32 audio to 16 kHz using linear interpolation.
   */
  _resampleTo16k(audio, sourceRate) {
    const targetRate = 16000;
    if (!sourceRate || sourceRate === targetRate) {
      return audio instanceof Float32Array ? audio : new Float32Array(audio);
    }

    const ratio = sourceRate / targetRate;
    const newLength = Math.round(audio.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
      const srcPos = i * ratio;
      const idx = Math.floor(srcPos);
      const frac = srcPos - idx;
      const a = audio[idx] || 0;
      const b = audio[idx + 1] !== undefined ? audio[idx + 1] : a;
      result[i] = a + (b - a) * frac;
    }

    return result;
  }
}

export default Transcriber;
