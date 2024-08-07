/// <reference types="@webgpu/types" />
// @ts-check

import { DataStream } from "./mp4box.js";
import * as MP4Box from "./mp4box.js";

class WebGPURenderer {
  /** @type {HTMLCanvasElement} */
  canvas;
  /** @type {GPUCanvasContext} */
  ctx;

  // WebGPU state shared between setup and drawing.
  format;
  /** @type {GPUDevice} */
  device;
  /** @type {GPURenderPipeline} */
  pipeline;
  /** @type {GPUSampler} */
  sampler;

  // Generates two triangles covering the whole canvas.
  static vertexShaderSource = `
    struct VertexOutput {
      @builtin(position) Position: vec4<f32>,
      @location(0) uv: vec2<f32>,
    }

    @vertex
    fn vert_main(@builtin(vertex_index) VertexIndex: u32) -> VertexOutput {
      var pos = array<vec2<f32>, 6>(
        vec2<f32>( 1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(-1.0,  1.0)
      );

      var uv = array<vec2<f32>, 6>(
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(0.0, 1.0),
        vec2<f32>(0.0, 0.0)
      );

      var output : VertexOutput;
      output.Position = vec4<f32>(pos[VertexIndex], 0.0, 1.0);
      output.uv = uv[VertexIndex];
      return output;
    }
  `;

  // Samples the external texture using generated UVs.
  static fragmentShaderSource = `
    @group(0) @binding(1) var mySampler: sampler;
    @group(0) @binding(2) var myTexture: texture_2d<f32>;
    
    @fragment
    fn frag_main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
      return textureSampleBaseClampToEdge(myTexture, mySampler, uv);
    }
  `;

  /** @param {HTMLCanvasElement} canvas */
  static async init(canvas) {
    const self = new WebGPURenderer();
    self.canvas = canvas;

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("failed to request adapter");
    }
    self.device = await adapter.requestDevice();
    self.format = navigator.gpu.getPreferredCanvasFormat();

    const ctx = self.canvas.getContext("webgpu");
    if (!ctx) {
      throw new Error("failed to get webgpu context");
    }
    self.ctx = ctx;
    self.ctx.configure({
      device: self.device,
      format: self.format,
      alphaMode: "opaque",
    });

    self.pipeline = self.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: self.device.createShaderModule({ code: WebGPURenderer.vertexShaderSource }),
        entryPoint: "vert_main",
      },
      fragment: {
        module: self.device.createShaderModule({ code: WebGPURenderer.fragmentShaderSource }),
        entryPoint: "frag_main",
        targets: [{ format: self.format }],
      },
      primitive: {
        topology: "triangle-list",
      },
    });

    // Default sampler configuration is nearset + clamp.
    self.sampler = self.device.createSampler({});

    return self;
  }

  /** @param {GPUTexture} frame */
  _resize(frame) {
    const targetWidth = 1024;
    const targetHeight = 576;

    let frameAspectRatio = frame.width / frame.height;
    let targetAspectRatio = targetWidth / targetHeight;

    let scale;
    if (frameAspectRatio > targetAspectRatio) {
      // image is wider relative to target
      scale = targetWidth / frame.width;
    } else {
      // image is taller relative to target
      scale = targetHeight / frame.height;
    }

    this.canvas.width = frame.width * scale;
    this.canvas.height = frame.height * scale;
  }

  /**
   *
   * @param {GPUTexture} frame
   */
  async draw(frame) {
    this._resize(frame);

    const uniformBindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: frame.createView() },
      ],
    });

    const commandEncoder = this.device.createCommandEncoder();
    const textureView = this.ctx.getCurrentTexture().createView();

    const passEncoder = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: [1.0, 0.0, 0.0, 1.0],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    passEncoder.setPipeline(this.pipeline);
    passEncoder.setBindGroup(0, uniformBindGroup);
    passEncoder.draw(6, 1, 0, 0);
    passEncoder.end();
    this.device.queue.submit([commandEncoder.finish()]);
  }
}

/** @param {string} url */
async function* fetchChunks(url) {
  const response = await fetch(url);

  if (!response.body) {
    throw new Error("body is empty");
  }

  const reader = response.body.getReader();
  try {
    let offset = 0;
    while (true) {
      const { done, value } = await reader.read();
      if (done) return;
      let currentOffset = offset;
      offset += value.byteLength;
      yield { offset: currentOffset, chunk: value };
    }
  } finally {
    reader.releaseLock();
  }
}

/** @param {string} url */
async function unboxVideo(url) {
  const mp4 = MP4Box.createFile();

  let track = {};
  let videoDecoderConfig = /** @type {VideoDecoderConfig} */ (/** @type {unknown} */ (undefined));
  let timescale = 1000;
  let duration = 0;

  mp4.onReady = (info) => {
    track = info.videoTracks[0];

    console.log(info);

    let description = null;
    const trak = mp4.getTrackById(track.id);
    for (const entry of trak.mdia.minf.stbl.stsd.entries) {
      const box = entry.avcC || entry.hvcC || entry.vpcC || entry.av1C;
      if (box) {
        const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
        box.write(stream);
        const buffer = /** @type {ArrayBuffer} */ (stream.buffer);
        description = new Uint8Array(buffer, 8); // Remove the box header.
        break;
      }
    }

    if (!description) {
      throw new Error("avcC, hvcC, vpcC, or av1C box not found");
    }

    videoDecoderConfig = {
      codec: track.codec.startsWith("vp08") ? "vp8" : track.codec,
      codedHeight: track.video.height,
      codedWidth: track.video.width,
      description,
    };
    timescale = info.timescale;
    duration = info.duration;

    mp4.setExtractionOptions(track.id);
    mp4.start();
  };

  let rawSamples = [];
  mp4.onSamples = (_a, _b, samples) => {
    Array.prototype.push.apply(rawSamples, samples);
  };

  for await (const { offset, chunk } of fetchChunks(url)) {
    mp4.appendBuffer(Object.assign(chunk.buffer, { fileStart: offset }));
  }
  mp4.flush();

  /** @type {Segment[]} */
  let segments = [];
  /** @type {Sample[]} */
  let samples = [];
  let data = new ArrayBuffer(mp4.samplesDataSize);
  let view = new Uint8Array(data);
  let byteOffset = 0;
  for (const sample of rawSamples) {
    let byteLength = sample.data.byteLength;
    view.set(sample.data, byteOffset);

    if (sample.is_sync) {
      if (samples.length !== 0) {
        segments.push({
          samples,
          start: samples[0].timestamp,
        });
        samples = [];
      }
    }

    samples.push({
      type: sample.is_sync ? "key" : "delta",
      timestamp: (1e6 * sample.cts) / sample.timescale,
      duration: (1e6 * sample.duration) / sample.timescale,
      byteOffset,
      byteLength,
    });

    byteOffset += byteLength;
  }

  if (samples.length !== 0) {
    segments.push({
      samples,
      start: samples[0].timestamp,
    });
  }

  if (!videoDecoderConfig) {
    // shouldn't happen, the callbacks are synchronous and should
    // be called by the time `flush` is called
    throw new Error("invalid ordering");
  }

  console.log(mp4);

  return {
    segments,
    videoDecoderConfig,
    timescale,
    duration,
    data,
  };
}

/**
 * @param {GPUDevice} device
 * @param {number} width
 * @param {number} height
 * @returns {GPUTexture}
 */
function allocVideoFrameTexture(device, width, height) {
  const texture = device.createTexture({
    label: `frame ${frames.length}`,
    size: {
      width,
      height,
      depthOrArrayLayers: 1,
    },
    mipLevelCount: 1,
    sampleCount: 1,
    dimension: "2d",
    format: "rgba8unorm",
    usage:
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.RENDER_ATTACHMENT,
  });
  return texture;
}

/**
 * @param {GPUDevice} device
 * @param {VideoFrame} frame
 * @param {GPUTexture} texture
 */
function copyVideoFrameToTexture(device, frame, texture) {
  let codedRect = frame.codedRect;
  if (!codedRect) {
    codedRect = new DOMRectReadOnly(0, 0, frame.codedWidth, frame.codedHeight);
  }

  device.queue.copyExternalImageToTexture(
    {
      source: frame,
      origin: {
        x: 0,
        y: 0,
      },
      flipY: false,
    },
    {
      texture,
      mipLevel: 0,
      origin: [0, 0, 0],
      aspect: "all",
      colorSpace: "srgb",
      premultipliedAlpha: false,
    },
    {
      width: frame.displayWidth,
      height: frame.displayHeight,
      depthOrArrayLayers: 1,
    }
  );
}

/**
 * @param {number} n
 * @param {number} min
 * @param {number} max
 */
function clamp(n, min, max) {
  return Math.min(Math.max(n, min), max);
}

/**
 * @template T, U
 * @param {T[]} arr
 * @param {U} target
 * @param {(v: T) => U} key
 * @returns {number}
 */
function latestAtIdx(arr, target, key) {
  let left = 0;
  let right = arr.length;

  if (arr.length === 0) {
    return -1;
  }

  if (arr.length > 0 && key(arr[0]) > target) {
    return -1;
  }

  while (left < right) {
    let middle = Math.floor((left + right) / 2);

    if (key(arr[middle]) <= target) {
      left = middle + 1;
    } else {
      right = middle;
    }
  }

  return left - 1;
}

/**
 * @template T, U
 * @param {T[]} arr
 * @param {U} target
 * @param {(v: T) => U} key
 * @returns {T | undefined}
 */
function latestAt(arr, target, key) {
  return arr[latestAtIdx(arr, target, key)];
}

/**
 * @typedef {{
 *   type: "key" | "delta",
 *   timestamp: number,
 *   duration: number,
 *   byteOffset: number,
 *   byteLength: number,
 * }} Sample
 *
 * @typedef {{
 *   samples: Sample[],
 *   start: number,
 * }} Segment
 */

class Video2 {
  /**
   * Loaded frames, always sorted in ascending order.
   *
   * @type {VideoFrame[]}
   */
  _frames = [];
  /** @type {VideoFrame | null} */
  _lastUsedFrame = null;

  /** @type {{ segment: number, sample: number }} */
  _currentPosition = { segment: -1000, sample: -1000 };

  static async load(/** @type {string} */ url) {
    const unboxed = await unboxVideo(url);
    console.log(unboxed);

    const { config, supported } = await VideoDecoder.isConfigSupported(unboxed.videoDecoderConfig);
    if (!supported) {
      throw new Error(
        `video decoder does not support config: ${JSON.stringify(unboxed.videoDecoderConfig, null, 2)}`
      );
    }

    return new Video2(
      renderer.device,
      unboxed.segments,
      config ?? unboxed.videoDecoderConfig,
      unboxed.duration,
      unboxed.timescale,
      unboxed.data
    );
  }

  constructor(
    /** @type {GPUDevice} */
    device,

    /** @type {Segment[]} */
    segments,
    /** @type {VideoDecoderConfig} */
    videoDecoderConfig,
    /** @type {number} */
    duration,
    /** @type {number} */
    timescale,
    /** @type {ArrayBuffer} */
    data
  ) {
    this.device = device;
    this.segments = segments;
    this.videoDecoderConfig = videoDecoderConfig;
    this.videoDecoder = new VideoDecoder({
      output: this._output,
      error(e) {
        console.error(e);
      },
    });
    this._duration = duration;
    this.timescale = timescale;
    this.data = data;
    /**
     * Start of the first segment.
     *
     * Segments do not need to start at timestamp 0,
     * and we need to handle that case by offsetting
     * all timestamps by the actual start timestamp
     * of the first segment.
     */
    this.timeOffset = segments[0].start;

    this._texture = allocVideoFrameTexture(
      this.device,
      /** @type {number} */ (this.videoDecoderConfig.codedWidth),
      /** @type {number} */ (this.videoDecoderConfig.codedHeight)
    );

    // immediately start buffering frames from the beginning of the video
    this._reset();
    this.getFrame(0);
  }

  /** Duration in timescale units */
  get duration() {
    return (this._duration * 1e6) / this.timescale;
  }

  /**
   * Get the latest frame at the given time.
   *
   * @param {number} timestamp in time units
   * @returns {GPUTexture | undefined}
   */
  getFrame(timestamp) {
    timestamp = timestamp + this.timeOffset;

    const newSegmentIdx = latestAtIdx(this.segments, timestamp, (s) => s.start);
    const newSampleIdx = latestAtIdx(
      this.segments[newSegmentIdx].samples,
      timestamp,
      (s) => s.timestamp
    );
    // NOTE: this check handles
    // - backward seek within a single segment, we already have those frames buffered
    // - forward seek past the duration of the video (timestamp > duration)
    // - backward seek past the start of the video (timestamp < 0)
    if (newSegmentIdx !== this._currentPosition.segment) {
      // buffer more samples from new segment
      // we maintain a buffer of all samples in segments `N` and `N+1`
      let segmentDistance = newSegmentIdx - this._currentPosition.segment;
      if (segmentDistance === 1) {
        // forward seek by 1
        this._enqueueAll(this.segments[newSegmentIdx + 1]);
      } else {
        // forward seek by N>1 OR backward seek across segments
        this._reset();
        this._enqueueAll(this.segments[newSegmentIdx]);
        this._enqueueAll(this.segments[newSegmentIdx + 1]);
      }
    } else if (newSampleIdx !== this._currentPosition.sample) {
      // segment index unchanged, but sample index did change
      const sampleDistance = newSampleIdx - this._currentPosition.sample;
      if (sampleDistance < 0) {
        // backward seek within a segment
        this._reset();
        this._enqueueAll(this.segments[newSegmentIdx]);
        this._enqueueAll(this.segments[newSegmentIdx + 1]);
      }
    }

    this._currentPosition.segment = newSegmentIdx;
    this._currentPosition.sample = newSampleIdx;

    const currentFrameIdx = latestAtIdx(this._frames, timestamp, (frame) => frame.timestamp);
    const currentFrame = this._frames[currentFrameIdx];
    if (!currentFrame) {
      // no buffered frames
      return;
    }

    // clear old frames so the decoder can output more
    for (const frame of this._frames.splice(0, currentFrameIdx)) {
      frame.close();
    }

    if (timestamp - currentFrame.timestamp > (currentFrame.duration ?? 0)) {
      // not relevant to the user, it's an old frame.
      return;
    }

    if (this._lastUsedFrame !== currentFrame) {
      copyVideoFrameToTexture(this.device, currentFrame, this._texture);
      this._lastUsedFrame = currentFrame;
    }
    return this._texture;
  }

  /**
   * Reset the video decoder and clear buffered frames.
   */
  _reset() {
    this.videoDecoder.reset();
    this.videoDecoder.configure(this.videoDecoderConfig);
    for (const frame of this._frames) {
      frame.close();
    }
    this._frames.length = 0;
  }

  /**
   * Give a full segment of video to the decoder.
   *
   * Segments always start with a `key` sample.
   */
  _enqueueAll(/** @type {Segment | undefined} */ segment) {
    if (!segment) return;
    console.log(segment);
    for (const sample of segment.samples) {
      this._enqueue(sample);
    }
  }

  /**
   * Give a sample to the decoder.
   *
   * ⚠ The first sample after a reset must have type `key`
   */
  _enqueue(/** @type {Sample} */ sample) {
    this.videoDecoder.decode(
      new EncodedVideoChunk({
        type: sample.type,
        timestamp: sample.timestamp,
        duration: sample.duration,
        data: this.data.slice(sample.byteOffset, sample.byteOffset + sample.byteLength),
      })
    );
  }

  /** Called by the video decoder when a full frame is ready. */
  _output = (/** @type {VideoFrame} */ frame) => {
    this._frames.push(frame);
  };
}

class VideoController {
  constructor(/** @type {Video2} */ video, /** @type {HTMLDivElement} */ controls) {
    this.video = video;
    this.button = /** @type {HTMLButtonElement} */ (controls.querySelector(".play"));
    this.container = /** @type {HTMLDivElement} */ (controls.querySelector(".timeline"));
    this.containerRect = this.container.getBoundingClientRect();
    this.currentTimeRect = /** @type {HTMLDivElement} */ (
      this.container.querySelector(".current-time")
    );

    /** How many milliseconds per frame */
    this.frameDuration = 1000 / 60;

    /** Current time in seconds */
    this.currentTime = 0;

    /**
     * Drag state
     *
     * @type {["start", number] | ["move", boolean] | null}
     */
    this.dragging = null;

    /**
     * Interval ID or null
     * @type {number | null}
     */
    this.playing = null;

    const setTime = (/** @type {number} */ timestamp_ts) => {
      // console.log(timestamp_ts);
      this.currentTime = timestamp_ts / video.timescale / 1e3;
      this.currentTimeRect.style.width = `${(timestamp_ts / video.duration) * 100}%`;
    };

    const onClick = (/** @type {number} */ clientX) => {
      const clampedX = clamp(clientX, this.containerRect.left, this.containerRect.right);
      const dist = clampedX - this.containerRect.left;
      const fract = dist / this.containerRect.width;
      const timestamp_ts = fract * video.duration;
      setTime(timestamp_ts);
    };

    const onPlaybackInterval = (/** @type {number} */ time_s) => {
      const timestamp_ts = time_s * video.timescale * 1e3;
      setTime(timestamp_ts);
    };

    this.container.addEventListener("pointerdown", (e) => {
      this.dragging = ["start", e.clientX];
    });
    window.addEventListener("pointermove", (e) => {
      if (!this.dragging) return;

      const [state, value] = this.dragging;
      switch (state) {
        case "start": {
          if (Math.abs(e.clientX - this.dragging[1]) > 4) {
            const wasPlaying = this.playing != null;
            pause();
            this.dragging = ["move", wasPlaying];
            onClick(e.clientX);
          }
          return;
        }
        case "move": {
          return onClick(e.clientX);
        }
      }
    });
    window.addEventListener("pointerup", (e) => {
      if (!this.dragging) return;

      const [state, value] = this.dragging;
      if (state !== "move") {
        onClick(e.clientX);
      } else {
        if (value) {
          play();
        }
      }

      this.dragging = null;
    });

    this.button.textContent = icons.play;
    const play = () => {
      console.log(this.frameDuration);
      this.button.textContent = icons.pause;
      this.playing = setInterval(
        () => onPlaybackInterval(this.currentTime + this.frameDuration / 1000),
        this.frameDuration
      );
    };
    const pause = () => {
      this.button.textContent = icons.play;
      clearInterval(this.playing ?? -1);
      this.playing = null;
    };
    this.button.addEventListener("click", () => {
      if (this.playing == null) {
        play();
      } else {
        pause();
      }
    });

    for (const segment of video.segments) {
      this._addTick(segment.start - video.timeOffset);
    }
  }

  get currentTimeTs() {
    return this.currentTime * this.video.timescale * 1e3;
  }

  /** @param {number} timestamp */
  _addTick(timestamp) {
    const pct = (timestamp / this.video.duration) * 100;
    const tick = document.createElement("div");
    tick.classList.add("tick");
    tick.style.width = `${pct}%`;

    this.container.appendChild(tick);
  }
}

const icons = {
  play: "▶",
  pause: "❚❚",
  loading: "⟳",
};

let renderer = await WebGPURenderer.init(
  /** @type {HTMLCanvasElement} */ (document.querySelector("canvas"))
);

let url = new URLSearchParams(window.location.search).get("url") ?? `data/bbb_video_av1_frag.mp4`;
const video = await Video2.load(url);

// @ts-expect-error
window._video = video;

// @ts-expect-error
const timeline = new VideoController(video, document.querySelector(".controls"));

function render() {
  const texture = video.getFrame(timeline.currentTimeTs);
  if (texture) renderer.draw(texture);
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// TODO: when loading a different video:
//       - the duration/timescale don't make sense
//       - wgpu is complaining about wrong copy rect sizes (???)
// TODO: can't seek to last frame
// TODO: playback

