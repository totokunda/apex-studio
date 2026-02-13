import { readFileBuffer } from "@app/preload";
import type { BaseClipApplicator } from "@/components/preview/clips/apply/base";
import type { FilterParams } from "@/components/preview/webgl-filters";
import {
  type ClipTransform,
  type FilterClipProps,
  type MaskClipProps,
  type MaskData,
  type MaskShapeTool,
} from "@/lib/types";

// This package's TS config does not include WebGPU DOM typings.
// Use lightweight local aliases + numeric usage flags to keep strict TS happy.
type GPUTexture = any;
type GPUDevice = any;
type GPUCanvasContext = any;
type GPUTextureFormat = any;
type GPUSampler = any;
type GPURenderPipeline = any;
type GPUBuffer = any;
type GPUCommandEncoder = any;
type GPUTextureView = any;
type GPURenderPassEncoder = any;
type GPUBindGroupLayout = any;

const GPUBufferUsage = {
  COPY_DST: 0x0008,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
};

const GPUTextureUsage = {
  COPY_DST: 0x02,
  TEXTURE_BINDING: 0x04,
  RENDER_ATTACHMENT: 0x10,
};

function getFragmentStageVisibility(): number {
  return (globalThis as any).GPUShaderStage?.FRAGMENT ?? 2;
}

const QUAD_WGSL = `
struct VsOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

@vertex
fn vsMain(@builtin(vertex_index) vertexIndex : u32) -> VsOut {
  var positions = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
  );
  var uvs = array<vec2<f32>, 4>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
  );
  var out: VsOut;
  out.position = vec4<f32>(positions[vertexIndex], 0.0, 1.0);
  out.uv = uvs[vertexIndex];
  return out;
}
`;

const FILTER_FROM_EXTERNAL_WGSL = `
${QUAD_WGSL}

struct FilterUniforms {
  p0: vec4<f32>, // width, height, brightness, contrast
  p1: vec4<f32>, // hue, saturation, blur, sharpness
  p2: vec4<f32>, // noise, vignette, tintIntensity, scanLines
  p3: vec4<f32>, // chroma, interlace, pixelate, jitter
  p4: vec4<f32>, // tintR, tintG, tintB, time
};

@group(0) @binding(0) var srcTex: texture_external;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: FilterUniforms;

fn hash2(p: vec2<f32>) -> f32 {
  let h = dot(p, vec2<f32>(127.1, 311.7));
  return fract(sin(h) * 43758.5453123);
}

fn rgb2hsv(c: vec3<f32>) -> vec3<f32> {
  let K = vec4<f32>(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
  let p = mix(vec4<f32>(c.bg, K.wz), vec4<f32>(c.gb, K.xy), select(0.0, 1.0, c.b < c.g));
  let q = mix(vec4<f32>(p.xyw, c.r), vec4<f32>(c.r, p.yzx), select(0.0, 1.0, p.x < c.r));
  let d = q.x - min(q.w, q.y);
  let e = 1e-10;
  return vec3<f32>(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

fn hue2rgb(p: f32, q: f32, t: f32) -> f32 {
  var tt = t;
  if (tt < 0.0) {
    tt = tt + 1.0;
  }
  if (tt > 1.0) {
    tt = tt - 1.0;
  }
  if (tt < (1.0 / 6.0)) {
    return p + (q - p) * 6.0 * tt;
  }
  if (tt < 0.5) {
    return q;
  }
  if (tt < (2.0 / 3.0)) {
    return p + (q - p) * ((2.0 / 3.0) - tt) * 6.0;
  }
  return p;
}

fn rgb2hsl(color: vec3<f32>) -> vec3<f32> {
  let maxC = max(max(color.r, color.g), color.b);
  let minC = min(min(color.r, color.g), color.b);
  let delta = maxC - minC;

  var h = 0.0;
  var s = 0.0;
  let l = (maxC + minC) * 0.5;

  if (delta > 0.0001) {
    if (l < 0.5) {
      s = delta / (maxC + minC);
    } else {
      s = delta / (2.0 - maxC - minC);
    }

    if (color.r >= maxC) {
      h = (color.g - color.b) / delta + select(0.0, 6.0, color.g < color.b);
    } else if (color.g >= maxC) {
      h = (color.b - color.r) / delta + 2.0;
    } else {
      h = (color.r - color.g) / delta + 4.0;
    }
    h = h / 6.0;
  }
  return vec3<f32>(h, s, l);
}

fn hsl2rgb(hsl: vec3<f32>) -> vec3<f32> {
  let h = hsl.x;
  let s = hsl.y;
  let l = hsl.z;
  if (s == 0.0) {
    return vec3<f32>(l, l, l);
  }
  let q = select(l * (1.0 + s), l + s - l * s, l >= 0.5);
  let p = 2.0 * l - q;
  return vec3<f32>(
    hue2rgb(p, q, h + (1.0 / 3.0)),
    hue2rgb(p, q, h),
    hue2rgb(p, q, h - (1.0 / 3.0)),
  );
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4<f32> {
  let width = max(uniforms.p0.x, 1.0);
  let height = max(uniforms.p0.y, 1.0);
  let uv = in.uv;
  let color = textureSampleBaseClampToEdge(srcTex, linearSampler, uv);
  var rgb = color.rgb;

  // Brightness, then contrast to mirror legacy filter ordering.
  rgb = rgb + uniforms.p0.z;
  rgb = (rgb - 0.5) * uniforms.p0.w + 0.5;

  // HSL-based hue/saturation to match legacy WebGL implementation.
  if (abs(uniforms.p1.x) > 0.0001 || abs(uniforms.p1.y - 1.0) > 0.0001) {
    var hsl = rgb2hsl(rgb);
    hsl.x = fract(hsl.x + uniforms.p1.x);
    hsl.y = clamp(hsl.y * uniforms.p1.y, 0.0, 1.0);
    rgb = hsl2rgb(hsl);
  }

  // Color tint uses luminance mapping, not channel multiply.
  if (uniforms.p4.x + uniforms.p4.y + uniforms.p4.z > 0.0001 && uniforms.p2.z > 0.001) {
    let tint = vec3<f32>(uniforms.p4.x, uniforms.p4.y, uniforms.p4.z);
    let luminance = dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
    let tinted = luminance * tint;
    rgb = mix(rgb, tinted, clamp(uniforms.p2.z, 0.0, 1.0));
  }

  // Vignette mirrors legacy cubic falloff.
  if (uniforms.p2.y > 0.001) {
    let centered = uv - vec2<f32>(0.5, 0.5);
    let dist = length(centered);
    let maxDist = length(vec2<f32>(0.5, 0.5));
    let normDist = dist / maxDist;
    let falloff = clamp(1.0 - (pow(normDist, 3.0) * uniforms.p2.y * 2.0), 0.0, 1.0);
    rgb = rgb * falloff;
  }

  // Scan lines with band modulation.
  if (uniforms.p2.w > 0.001) {
    let y = (1.0 - uv.y) * height;
    let lineSpacing = 3.0;
    let scanLine = 0.5 + 0.5 * sin(y * 3.14159265 / (lineSpacing * 0.5));
    let dimFactor = mix(1.0, scanLine, uniforms.p2.w * 0.6);
    let largeBand = 0.5 + 0.5 * sin(y * 3.14159265 / (height * 0.25));
    let bandFactor = mix(1.0, 0.92 + 0.08 * largeBand, uniforms.p2.w);
    rgb = rgb * dimFactor * bandFactor;
  }

  return vec4<f32>(rgb, color.a);
}
`;

const MASK_PASS_WGSL = `
${QUAD_WGSL}

struct MaskUniforms {
  p0: vec4<f32>, // canvasW, canvasH, toolKind, inverted
  p1: vec4<f32>, // backgroundEnabled, maskEnabled, backgroundOpacity, maskOpacity
  p2: vec4<f32>, // backgroundColor rgb, shapeKind
  p3: vec4<f32>, // maskColor rgb, contourCount
  p4: vec4<f32>, // shape x,y,w,h
  p5: vec4<f32>, // shapeScaleX, shapeScaleY, rotation, pointCount
};

struct Point {
  pos: vec2<f32>,
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;
@group(0) @binding(2) var<uniform> uniforms: MaskUniforms;
@group(0) @binding(3) var<storage, read> points: array<Point>;
@group(0) @binding(4) var<storage, read> contours: array<vec2<u32>>;

fn pointInTriangle(p: vec2<f32>, a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> bool {
  let v0 = c - a;
  let v1 = b - a;
  let v2 = p - a;
  let dot00 = dot(v0, v0);
  let dot01 = dot(v0, v1);
  let dot02 = dot(v0, v2);
  let dot11 = dot(v1, v1);
  let dot12 = dot(v1, v2);
  let denom = dot00 * dot11 - dot01 * dot01;
  if (abs(denom) < 0.000001) {
    return false;
  }
  let inv = 1.0 / denom;
  let u = (dot11 * dot02 - dot01 * dot12) * inv;
  let v = (dot00 * dot12 - dot01 * dot02) * inv;
  return u >= 0.0 && v >= 0.0 && (u + v) <= 1.0;
}

fn pointInContour(p: vec2<f32>, contourIndex: u32) -> bool {
  let contour = contours[contourIndex];
  let startIdx = contour.x;
  let count = contour.y;
  if (count < 3u) {
    return false;
  }

  var inside = false;
  var prev = points[startIdx + count - 1u].pos;
  var i: u32 = 0u;
  loop {
    if (i >= count || i >= 8192u) {
      break;
    }
    let cur = points[startIdx + i].pos;
    let cond = (cur.y > p.y) != (prev.y > p.y);
    if (cond) {
      let xIntersect = (prev.x - cur.x) * (p.y - cur.y) / ((prev.y - cur.y) + 0.000001) + cur.x;
      if (p.x < xIntersect) {
        inside = !inside;
      }
    }
    prev = cur;
    i = i + 1u;
  }
  return inside;
}

fn pointInAnyContour(p: vec2<f32>) -> bool {
  let contourCount = u32(max(uniforms.p3.w, 0.0));
  var i: u32 = 0u;
  loop {
    if (i >= contourCount || i >= 2048u) {
      break;
    }
    if (pointInContour(p, i)) {
      return true;
    }
    i = i + 1u;
  }
  return false;
}

fn pointInShape(pixelCoord: vec2<f32>) -> bool {
  let shapeX = uniforms.p4.x;
  let shapeY = uniforms.p4.y;
  let shapeW = max(uniforms.p4.z, 0.0001);
  let shapeH = max(uniforms.p4.w, 0.0001);
  let scaleX = max(uniforms.p5.x, 0.0001);
  let scaleY = max(uniforms.p5.y, 0.0001);
  let rotation = uniforms.p5.z;
  let shapeKind = i32(round(uniforms.p2.w));

  let w = shapeW * scaleX;
  let h = shapeH * scaleY;
  let halfW = w * 0.5;
  let halfH = h * 0.5;

  let c = cos(-rotation);
  let s = sin(-rotation);

  if (shapeKind == 0) {
    let pivot = vec2<f32>(shapeX, shapeY);
    let local = pixelCoord - pivot;
    let rot = vec2<f32>(c * local.x - s * local.y, s * local.x + c * local.y);
    return rot.x >= 0.0 && rot.x <= w && rot.y >= 0.0 && rot.y <= h;
  }

  let center = vec2<f32>(shapeX + 0.5 * w, shapeY + 0.5 * h);
  let local = pixelCoord - center;
  let rot = vec2<f32>(c * local.x - s * local.y, s * local.x + c * local.y);

  if (shapeKind == 1) {
    let radii = vec2<f32>(max(halfW, 0.0001), max(halfH, 0.0001));
    let d = rot / radii;
    return dot(d, d) <= 1.0;
  }

  if (shapeKind == 2) {
    let yOffset = -h / 6.0;
    let apex = vec2<f32>(0.0, -halfH + yOffset);
    let left = vec2<f32>(-halfW, halfH + yOffset);
    let right = vec2<f32>(halfW, halfH + yOffset);
    return pointInTriangle(rot, apex, left, right);
  }

  var inside = false;
  let rOuterX = max(halfW, 0.0001);
  let rOuterY = max(halfH, 0.0001);
  let rInnerX = 0.5 * rOuterX;
  let rInnerY = 0.5 * rOuterY;
  var prevVertex = vec2<f32>(
    rInnerX * sin(9.0 * 3.14159265358979323846 / 5.0),
    -rInnerY * cos(9.0 * 3.14159265358979323846 / 5.0),
  );
  var useOuter = true;
  var i: i32 = 0;
  loop {
    if (i >= 10) {
      break;
    }
    let rx = select(rInnerX, rOuterX, useOuter);
    let ry = select(rInnerY, rOuterY, useOuter);
    let angle = f32(i) * 3.14159265358979323846 / 5.0;
    let vi = vec2<f32>(rx * sin(angle), -ry * cos(angle));
    let intersect = ((vi.y > rot.y) != (prevVertex.y > rot.y)) &&
      (rot.x < (prevVertex.x - vi.x) * (rot.y - vi.y) / ((prevVertex.y - vi.y) + 0.000001) + vi.x);
    if (intersect) {
      inside = !inside;
    }
    prevVertex = vi;
    useOuter = !useOuter;
    i = i + 1;
  }
  return inside;
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4<f32> {
  let src = textureSample(srcTex, linearSampler, in.uv);
  let canvasW = max(uniforms.p0.x, 1.0);
  let canvasH = max(uniforms.p0.y, 1.0);
  let pixelCoord = vec2<f32>(in.uv.x * canvasW, (1.0 - in.uv.y) * canvasH);

  let toolKind = i32(round(uniforms.p0.z));
  var inside = false;
  if (toolKind == 0) {
    inside = pointInShape(pixelCoord);
  } else {
    inside = pointInAnyContour(pixelCoord);
  }

  if (uniforms.p0.w > 0.5) {
    inside = !inside;
  }

  let backgroundEnabled = uniforms.p1.x > 0.5;
  let maskEnabled = uniforms.p1.y > 0.5;
  let bgOpacity = clamp(uniforms.p1.z, 0.0, 1.0);
  let maskOpacity = clamp(uniforms.p1.w, 0.0, 1.0);
  let bgColor = uniforms.p2.rgb;
  let maskColor = uniforms.p3.rgb;

  if (inside) {
    if (maskEnabled) {
      return vec4<f32>(maskColor, maskOpacity);
    }
    return src;
  }

  if (backgroundEnabled) {
    return vec4<f32>(bgColor, bgOpacity);
  }
  return src;
}
`;

const CLUT_PASS_WGSL = `
${QUAD_WGSL}

struct ClutUniforms {
  p0: vec4<f32>, // clutSize, clutWidth, strength, pad
};

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSampler: sampler;
@group(0) @binding(2) var clutTex: texture_2d<f32>;
@group(0) @binding(3) var clutSampler: sampler;
@group(0) @binding(4) var<uniform> uniforms: ClutUniforms;

fn getHaldCoord(rgb: vec3<f32>, level: f32, cubeSize: f32, width: f32) -> vec2<f32> {
  let offset = rgb.x + level * rgb.y + cubeSize * rgb.z;
  let x = offset - floor(offset / width) * width;
  let y = floor(offset / width);
  return (vec2<f32>(x, y) + vec2<f32>(0.5, 0.5)) / width;
}

fn applyHald(color: vec3<f32>) -> vec3<f32> {
  let level = uniforms.p0.x;
  let levelMinusOne = max(level - 1.0, 1.0);
  let cubeSize = level * level;
  let width = uniforms.p0.y;
  let c = clamp(color, vec3<f32>(0.0), vec3<f32>(1.0));
  let scaled = c * levelMinusOne;
  let baseColor = floor(scaled);
  let nextColor = min(baseColor + vec3<f32>(1.0), vec3<f32>(levelMinusOne));
  let frac = scaled - baseColor;

  let c000 = textureSample(clutTex, clutSampler, getHaldCoord(vec3<f32>(baseColor.x, baseColor.y, baseColor.z), level, cubeSize, width)).rgb;
  let c001 = textureSample(clutTex, clutSampler, getHaldCoord(vec3<f32>(baseColor.x, baseColor.y, nextColor.z), level, cubeSize, width)).rgb;
  let c010 = textureSample(clutTex, clutSampler, getHaldCoord(vec3<f32>(baseColor.x, nextColor.y, baseColor.z), level, cubeSize, width)).rgb;
  let c011 = textureSample(clutTex, clutSampler, getHaldCoord(vec3<f32>(baseColor.x, nextColor.y, nextColor.z), level, cubeSize, width)).rgb;
  let c100 = textureSample(clutTex, clutSampler, getHaldCoord(vec3<f32>(nextColor.x, baseColor.y, baseColor.z), level, cubeSize, width)).rgb;
  let c101 = textureSample(clutTex, clutSampler, getHaldCoord(vec3<f32>(nextColor.x, baseColor.y, nextColor.z), level, cubeSize, width)).rgb;
  let c110 = textureSample(clutTex, clutSampler, getHaldCoord(vec3<f32>(nextColor.x, nextColor.y, baseColor.z), level, cubeSize, width)).rgb;
  let c111 = textureSample(clutTex, clutSampler, getHaldCoord(vec3<f32>(nextColor.x, nextColor.y, nextColor.z), level, cubeSize, width)).rgb;

  let c00 = mix(c000, c100, frac.x);
  let c01 = mix(c001, c101, frac.x);
  let c10 = mix(c010, c110, frac.x);
  let c11 = mix(c011, c111, frac.x);
  let c0 = mix(c00, c10, frac.y);
  let c1 = mix(c01, c11, frac.y);
  return mix(c0, c1, frac.z);
}

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4<f32> {
  let src = textureSample(srcTex, srcSampler, in.uv);
  let transformed = applyHald(src.rgb);
  let strength = clamp(uniforms.p0.z, 0.0, 1.0);
  let rgb = mix(src.rgb, transformed, strength);
  return vec4<f32>(rgb, src.a);
}
`;

const PRESENT_PASS_WGSL = `
${QUAD_WGSL}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var linearSampler: sampler;

@fragment
fn fsMain(in: VsOut) -> @location(0) vec4<f32> {
  return textureSample(srcTex, linearSampler, in.uv);
}
`;

type ClutTextureInfo = {
  texture: GPUTexture;
  width: number;
  size: number;
};

type ActiveFilterApplicator = {
  path: string;
  strength: number;
};

type NormalizedMaskPass = {
  toolKind: number;
  inverted: number;
  backgroundEnabled: number;
  maskEnabled: number;
  backgroundColor: [number, number, number];
  backgroundOpacity: number;
  maskColor: [number, number, number];
  maskOpacity: number;
  shapeKind: number;
  shapeBounds: [number, number, number, number];
  shapeScale: [number, number];
  rotation: number;
  points: Float32Array;
  contours: Uint32Array;
};

function getGpuApi(): any | null {
  return (navigator as any)?.gpu ?? null;
}

let sharedPreviewDevicePromise: Promise<any | null> | null = null;

async function getSharedPreviewDevice(): Promise<any | null> {
  if (sharedPreviewDevicePromise) {
    return sharedPreviewDevicePromise;
  }
  sharedPreviewDevicePromise = (async () => {
    const gpu = getGpuApi();
    if (!gpu?.requestAdapter) return null;
    try {
      const adapter = await gpu.requestAdapter();
      if (!adapter) return null;
      return await adapter.requestDevice();
    } catch {
      return null;
    }
  })();
  return sharedPreviewDevicePromise;
}

async function shaderModuleHasErrors(module: any): Promise<boolean> {
  if (!module?.getCompilationInfo) return false;
  try {
    const info = await module.getCompilationInfo();
    const messages = info?.messages ?? [];
    return messages.some((m: any) => m?.type === "error");
  } catch {
    return false;
  }
}

async function createValidatedRenderPipeline(device: any, descriptor: any): Promise<any> {
  const canScopeErrors =
    typeof device?.pushErrorScope === "function" && typeof device?.popErrorScope === "function";
  if (canScopeErrors) {
    device.pushErrorScope("validation");
  }
  const pipeline = await device.createRenderPipelineAsync(descriptor);
  if (canScopeErrors) {
    const error = await device.popErrorScope();
    if (error) {
      throw error;
    }
  }
  return pipeline;
}

function parseHexColor(color: string | undefined, fallback: [number, number, number]): [number, number, number] {
  if (!color) return fallback;
  const match = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(color);
  if (!match) return fallback;
  return [
    parseInt(match[1], 16) / 255,
    parseInt(match[2], 16) / 255,
    parseInt(match[3], 16) / 255,
  ];
}

function shapeKindToNumber(shape: MaskShapeTool | undefined): number {
  if (shape === "ellipse") return 1;
  if (shape === "polygon") return 2;
  if (shape === "star") return 3;
  return 0;
}

function resolveMaskData(mask: MaskClipProps, frame: number): MaskData | null {
  const keyframes = mask.keyframes;
  const entries =
    keyframes instanceof Map
      ? Array.from(keyframes.entries()).map(([k, v]) => [Number(k), v] as const)
      : Object.keys(keyframes).map((k) => [Number(k), (keyframes as Record<number, MaskData>)[Number(k)]] as const);
  if (entries.length === 0) return null;
  entries.sort((a, b) => a[0] - b[0]);
  let selected = entries[0][1];
  for (const [k, v] of entries) {
    if (k <= frame) {
      selected = v;
    } else {
      break;
    }
  }
  return selected;
}

function computeCanvasScale(
  canvasWidth: number,
  canvasHeight: number,
  clipTransform?: ClipTransform,
  maskTransform?: ClipTransform,
): { scaleX: number; scaleY: number } {
  const baseScaleX = clipTransform?.scaleX ?? maskTransform?.scaleX ?? 1;
  const baseScaleY = clipTransform?.scaleY ?? maskTransform?.scaleY ?? 1;
  const baseWidth = (clipTransform?.width ?? maskTransform?.width ?? canvasWidth) * baseScaleX;
  const baseHeight = (clipTransform?.height ?? maskTransform?.height ?? canvasHeight) * baseScaleY;
  return {
    scaleX: baseWidth !== 0 ? canvasWidth / baseWidth : 1,
    scaleY: baseHeight !== 0 ? canvasHeight / baseHeight : 1,
  };
}

function applyShapeClipTransform(
  shapeBounds: {
    x: number;
    y: number;
    width: number;
    height: number;
    scaleX?: number;
    scaleY?: number;
    rotation?: number;
    shapeType?: MaskShapeTool;
  },
  originalClipTransform?: ClipTransform,
  clipTransform?: ClipTransform,
  maskTransform?: ClipTransform,
): {
  x: number;
  y: number;
  width: number;
  height: number;
  scaleX: number;
  scaleY: number;
  rotation: number;
  shapeType?: MaskShapeTool;
} {
  if (!clipTransform) {
    return {
      x: shapeBounds.x,
      y: shapeBounds.y,
      width: shapeBounds.width,
      height: shapeBounds.height,
      scaleX: shapeBounds.scaleX ?? 1,
      scaleY: shapeBounds.scaleY ?? 1,
      rotation: shapeBounds.rotation ?? 0,
      shapeType: shapeBounds.shapeType,
    };
  }

  let localX: number;
  let localY: number;
  let scaledWidth: number;
  let scaledHeight: number;

  if (maskTransform) {
    const baseScaleX = maskTransform.scaleX || 1;
    const baseScaleY = maskTransform.scaleY || 1;
    const scaleRatioX = (clipTransform.scaleX || 1) / baseScaleX;
    const scaleRatioY = (clipTransform.scaleY || 1) / baseScaleY;
    let deltaX = maskTransform.x - (originalClipTransform?.x ?? 0);
    let deltaY = maskTransform.y - (originalClipTransform?.y ?? 0);

    if (maskTransform.crop) {
      const fullWidth = maskTransform.width / maskTransform.crop.width;
      const fullHeight = maskTransform.height / maskTransform.crop.height;
      deltaX -= fullWidth * maskTransform.crop.x;
      deltaY -= fullHeight * maskTransform.crop.y;
    }

    if (originalClipTransform) {
      localX = (shapeBounds.x - maskTransform.x + deltaX) * scaleRatioX;
      localY = (shapeBounds.y - maskTransform.y + deltaY) * scaleRatioY;
    } else {
      localX = shapeBounds.x * scaleRatioX;
      localY = shapeBounds.y * scaleRatioY;
    }
    scaledWidth = shapeBounds.width * scaleRatioX;
    scaledHeight = shapeBounds.height * scaleRatioY;
  } else {
    localX = shapeBounds.x;
    localY = shapeBounds.y;
    scaledWidth = shapeBounds.width;
    scaledHeight = shapeBounds.height;
  }

  if (clipTransform.crop) {
    const displayWidth = Math.abs((clipTransform.width || 0) * (clipTransform.scaleX || 1));
    const displayHeight = Math.abs((clipTransform.height || 0) * (clipTransform.scaleY || 1));
    localX = clipTransform.crop.x * displayWidth + localX * clipTransform.crop.width;
    localY = clipTransform.crop.y * displayHeight + localY * clipTransform.crop.height;
    scaledWidth *= clipTransform.crop.width;
    scaledHeight *= clipTransform.crop.height;
  }

  return {
    x: localX,
    y: localY,
    width: scaledWidth,
    height: scaledHeight,
    scaleX: shapeBounds.scaleX ?? 1,
    scaleY: shapeBounds.scaleY ?? 1,
    rotation: shapeBounds.rotation ?? 0,
    shapeType: shapeBounds.shapeType,
  };
}

function applyLassoClipTransform(
  lassoPoints: number[],
  originalClipTransform?: ClipTransform,
  clipTransform?: ClipTransform,
  maskTransform?: ClipTransform,
): number[] {
  if (!clipTransform) return lassoPoints;
  let scaleRatioX = 1;
  let scaleRatioY = 1;
  let deltaX = 0;
  let deltaY = 0;

  if (maskTransform) {
    const baseScaleX = maskTransform.scaleX || 1;
    const baseScaleY = maskTransform.scaleY || 1;
    scaleRatioX = (clipTransform.scaleX || 1) / baseScaleX;
    scaleRatioY = (clipTransform.scaleY || 1) / baseScaleY;
    deltaX = maskTransform.x - (originalClipTransform?.x ?? 0);
    deltaY = maskTransform.y - (originalClipTransform?.y ?? 0);
    if (maskTransform.crop) {
      const fullWidth = (maskTransform.width || 0) / (maskTransform.crop.width || 1);
      const fullHeight = (maskTransform.height || 0) / (maskTransform.crop.height || 1);
      deltaX -= fullWidth * (maskTransform.crop.x || 0);
      deltaY -= fullHeight * (maskTransform.crop.y || 0);
    }
  }

  const out: number[] = [];
  const hasCrop = !!clipTransform.crop;
  const displayWidth = Math.abs((clipTransform.width || 0) * (clipTransform.scaleX || 1));
  const displayHeight = Math.abs((clipTransform.height || 0) * (clipTransform.scaleY || 1));

  for (let i = 0; i < lassoPoints.length; i += 2) {
    let x = maskTransform ? (lassoPoints[i] - maskTransform.x + deltaX) * scaleRatioX : lassoPoints[i];
    let y = maskTransform ? (lassoPoints[i + 1] - maskTransform.y + deltaY) * scaleRatioY : lassoPoints[i + 1];
    if (hasCrop && clipTransform.crop) {
      x = clipTransform.crop.x * displayWidth + x * clipTransform.crop.width;
      y = clipTransform.crop.y * displayHeight + y * clipTransform.crop.height;
    }
    out.push(x, y);
  }
  return out;
}

function applyTouchClipTransform(
  contours: number[][],
  originalClipTransform?: ClipTransform,
  clipTransform?: ClipTransform,
  maskTransform?: ClipTransform,
): number[][] {
  if (!clipTransform || !maskTransform) return contours;
  const baseScaleX = maskTransform.scaleX || 1;
  const baseScaleY = maskTransform.scaleY || 1;
  const scaleRatioX = (clipTransform.scaleX || 1) / baseScaleX;
  const scaleRatioY = (clipTransform.scaleY || 1) / baseScaleY;
  let deltaX = maskTransform.x - (originalClipTransform?.x ?? 0);
  let deltaY = maskTransform.y - (originalClipTransform?.y ?? 0);
  if (maskTransform.crop) {
    const fullWidth = (maskTransform.width || 0) / (maskTransform.crop.width || 1);
    const fullHeight = (maskTransform.height || 0) / (maskTransform.crop.height || 1);
    deltaX -= fullWidth * (maskTransform.crop.x || 0);
    deltaY -= fullHeight * (maskTransform.crop.y || 0);
  }

  const hasCrop = !!clipTransform.crop;
  const displayWidth = Math.abs((clipTransform.width || 0) * (clipTransform.scaleX || 1));
  const displayHeight = Math.abs((clipTransform.height || 0) * (clipTransform.scaleY || 1));
  const out: number[][] = [];

  for (const contour of contours) {
    const next: number[] = [];
    for (let i = 0; i < contour.length; i += 2) {
      let x = (contour[i] - maskTransform.x + deltaX) * scaleRatioX;
      let y = (contour[i + 1] - maskTransform.y + deltaY) * scaleRatioY;
      if (hasCrop && clipTransform.crop) {
        x = clipTransform.crop.x * displayWidth + x * clipTransform.crop.width;
        y = clipTransform.crop.y * displayHeight + y * clipTransform.crop.height;
      }
      next.push(x, y);
    }
    out.push(next);
  }
  return out;
}

function normalizeMaskPass(
  mask: MaskClipProps,
  frame: number,
  canvasWidth: number,
  canvasHeight: number,
  clipTransform?: ClipTransform,
  originalClipTransform?: ClipTransform,
  isFirstMask = false,
): NormalizedMaskPass | null {
  const keyData = resolveMaskData(mask, frame);
  if (!keyData) return null;

  const backgroundColor = parseHexColor(mask.backgroundColor, [0, 0, 0]);
  const maskColor = parseHexColor(mask.maskColor, [1, 1, 1]);
  const backgroundOpacity = Math.max(0, Math.min(1, (mask.backgroundOpacity ?? 100) / 100));
  const maskOpacity = Math.max(0, Math.min(1, (mask.maskOpacity ?? 100) / 100));

  const basePass: NormalizedMaskPass = {
    toolKind: mask.tool === "shape" ? 0 : mask.tool === "lasso" ? 1 : 2,
    inverted: mask.inverted ? 1 : 0,
    backgroundEnabled: isFirstMask && mask.backgroundColorEnabled ? 1 : 0,
    maskEnabled: mask.maskColorEnabled ? 1 : 0,
    backgroundColor,
    backgroundOpacity,
    maskColor,
    maskOpacity,
    shapeKind: 0,
    shapeBounds: [0, 0, 0, 0],
    shapeScale: [1, 1],
    rotation: 0,
    points: new Float32Array(0),
    contours: new Uint32Array(0),
  };

  const maskTransform = mask.transform;
  const { scaleX, scaleY } = computeCanvasScale(
    canvasWidth,
    canvasHeight,
    clipTransform,
    maskTransform,
  );

  if (mask.tool === "shape" && keyData.shapeBounds) {
    const transformed = applyShapeClipTransform(
      keyData.shapeBounds,
      originalClipTransform,
      clipTransform,
      maskTransform,
    );
    basePass.shapeKind = shapeKindToNumber(transformed.shapeType);
    basePass.shapeBounds = [
      transformed.x * scaleX,
      transformed.y * scaleY,
      transformed.width * scaleX,
      transformed.height * scaleY,
    ];
    basePass.shapeScale = [transformed.scaleX, transformed.scaleY];
    basePass.rotation = (transformed.rotation ?? 0) * (Math.PI / 180);
    return basePass;
  }

  if (mask.tool === "lasso" && keyData.lassoPoints && keyData.lassoPoints.length >= 6) {
    const transformed = applyLassoClipTransform(
      keyData.lassoPoints,
      originalClipTransform,
      clipTransform,
      maskTransform,
    );
    const points = new Float32Array(transformed.length);
    for (let i = 0; i < transformed.length; i += 2) {
      points[i] = transformed[i] * scaleX;
      points[i + 1] = transformed[i + 1] * scaleY;
    }
    basePass.points = points;
    basePass.contours = new Uint32Array([0, transformed.length / 2]);
    return basePass;
  }

  if (mask.tool === "touch" && keyData.contours && keyData.contours.length > 0) {
    const transformedContours = applyTouchClipTransform(
      keyData.contours,
      originalClipTransform,
      clipTransform,
      maskTransform,
    );
    const pointsAcc: number[] = [];
    const contourRanges: number[] = [];
    let start = 0;
    for (const contour of transformedContours) {
      const count = Math.floor(contour.length / 2);
      if (count < 3) continue;
      contourRanges.push(start, count);
      for (let i = 0; i < contour.length; i += 2) {
        pointsAcc.push(contour[i] * scaleX, contour[i + 1] * scaleY);
      }
      start += count;
    }
    if (start < 3) return null;
    basePass.points = new Float32Array(pointsAcc);
    basePass.contours = new Uint32Array(contourRanges);
    return basePass;
  }

  return null;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function extractActiveFilterApplicators(
  applicators: BaseClipApplicator[],
  focusFrame: number,
): ActiveFilterApplicator[] {
  const out: ActiveFilterApplicator[] = [];
  for (const applicator of applicators) {
    const maybeClip = (applicator as any)?.getClip?.() as FilterClipProps | undefined;
    if (!maybeClip || maybeClip.type !== "filter") continue;
    const start = maybeClip.startFrame ?? 0;
    const end = maybeClip.endFrame ?? Number.MAX_SAFE_INTEGER;
    if (focusFrame < start || focusFrame > end) continue;
    const path = maybeClip.fullPath || maybeClip.smallPath;
    if (!path) continue;
    const maybeStrength = (applicator as any)?.getIntensity?.();
    let strength = typeof maybeStrength === "number" ? maybeStrength : maybeClip.intensity ?? 100;
    if (strength > 1) {
      strength = strength / 100;
    }
    out.push({
      path,
      strength: clamp01(strength),
    });
  }
  return out;
}

function extractAllFilterApplicatorPaths(applicators: BaseClipApplicator[]): string[] {
  const paths: string[] = [];
  for (const applicator of applicators) {
    const maybeClip = (applicator as any)?.getClip?.() as FilterClipProps | undefined;
    if (!maybeClip || maybeClip.type !== "filter") continue;
    const path = maybeClip.fullPath || maybeClip.smallPath;
    if (!path) continue;
    paths.push(path);
  }
  return paths;
}

export class WebGPUPreviewPipeline {
  private readonly canvas: HTMLCanvasElement;
  private readonly device: GPUDevice;
  private readonly context: GPUCanvasContext;
  private readonly canvasFormat: GPUTextureFormat;
  private readonly workingFormat: GPUTextureFormat = "rgba8unorm";
  private readonly linearSampler: GPUSampler;
  private readonly nearestSampler: GPUSampler;
  private readonly filterBindGroupLayout: GPUBindGroupLayout;
  private readonly maskBindGroupLayout: GPUBindGroupLayout;
  private readonly clutBindGroupLayout: GPUBindGroupLayout;
  private readonly presentBindGroupLayout: GPUBindGroupLayout;

  private filterPipeline: GPURenderPipeline;
  private maskPipeline: GPURenderPipeline;
  private clutPipeline: GPURenderPipeline;
  private presentPipeline: GPURenderPipeline;

  private filterUniformBuffer: GPUBuffer;
  private maskUniformBuffer: GPUBuffer;
  private clutUniformBuffer: GPUBuffer;
  private dummyPointsBuffer: GPUBuffer;
  private dummyContoursBuffer: GPUBuffer;

  private pingTextureA: GPUTexture | null = null;
  private pingTextureB: GPUTexture | null = null;
  private pingWidth = 0;
  private pingHeight = 0;

  private clutCache = new Map<string, ClutTextureInfo>();
  private clutLoading = new Map<string, Promise<void>>();

  private constructor(
    canvas: HTMLCanvasElement,
    device: GPUDevice,
    context: GPUCanvasContext,
    canvasFormat: GPUTextureFormat,
    filterPipeline: GPURenderPipeline,
    maskPipeline: GPURenderPipeline,
    clutPipeline: GPURenderPipeline,
    presentPipeline: GPURenderPipeline,
    filterBindGroupLayout: GPUBindGroupLayout,
    maskBindGroupLayout: GPUBindGroupLayout,
    clutBindGroupLayout: GPUBindGroupLayout,
    presentBindGroupLayout: GPUBindGroupLayout,
  ) {
    this.canvas = canvas;
    this.device = device;
    this.context = context;
    this.canvasFormat = canvasFormat;
    this.filterPipeline = filterPipeline;
    this.maskPipeline = maskPipeline;
    this.clutPipeline = clutPipeline;
    this.presentPipeline = presentPipeline;
    this.filterBindGroupLayout = filterBindGroupLayout;
    this.maskBindGroupLayout = maskBindGroupLayout;
    this.clutBindGroupLayout = clutBindGroupLayout;
    this.presentBindGroupLayout = presentBindGroupLayout;
    this.linearSampler = device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
    });
    this.nearestSampler = device.createSampler({
      magFilter: "nearest",
      minFilter: "nearest",
    });
    this.filterUniformBuffer = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.maskUniformBuffer = device.createBuffer({
      size: 96,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.clutUniformBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.dummyPointsBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.dummyContoursBuffer = device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
  }

  static async create(canvas: HTMLCanvasElement): Promise<WebGPUPreviewPipeline | null> {
    const gpu = getGpuApi();
    if (!gpu?.requestAdapter) return null;
    try {
      const device = await getSharedPreviewDevice();
      if (!device) return null;
      const context = canvas.getContext("webgpu") as any;
      if (!context) return null;
      const canvasFormat = gpu.getPreferredCanvasFormat
        ? gpu.getPreferredCanvasFormat()
        : "bgra8unorm";
      context.configure({
        device,
        format: canvasFormat,
        alphaMode: "premultiplied",
      });

      const filterModule = device.createShaderModule({ code: FILTER_FROM_EXTERNAL_WGSL });
      const maskModule = device.createShaderModule({ code: MASK_PASS_WGSL });
      const clutModule = device.createShaderModule({ code: CLUT_PASS_WGSL });
      const presentModule = device.createShaderModule({ code: PRESENT_PASS_WGSL });
      const hasShaderErrors = (
        await Promise.all([
          shaderModuleHasErrors(filterModule),
          shaderModuleHasErrors(maskModule),
          shaderModuleHasErrors(clutModule),
          shaderModuleHasErrors(presentModule),
        ])
      ).some(Boolean);
      if (hasShaderErrors) {
        return null;
      }

      const fragmentVisibility = getFragmentStageVisibility();
      const filterBindGroupLayout = device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility: fragmentVisibility, externalTexture: {} },
          { binding: 1, visibility: fragmentVisibility, sampler: { type: "filtering" } },
          { binding: 2, visibility: fragmentVisibility, buffer: { type: "uniform" } },
        ],
      });
      const maskBindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: fragmentVisibility,
            texture: { sampleType: "float", viewDimension: "2d" },
          },
          { binding: 1, visibility: fragmentVisibility, sampler: { type: "filtering" } },
          { binding: 2, visibility: fragmentVisibility, buffer: { type: "uniform" } },
          { binding: 3, visibility: fragmentVisibility, buffer: { type: "read-only-storage" } },
          { binding: 4, visibility: fragmentVisibility, buffer: { type: "read-only-storage" } },
        ],
      });
      const clutBindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: fragmentVisibility,
            texture: { sampleType: "float", viewDimension: "2d" },
          },
          { binding: 1, visibility: fragmentVisibility, sampler: { type: "filtering" } },
          {
            binding: 2,
            visibility: fragmentVisibility,
            texture: { sampleType: "float", viewDimension: "2d" },
          },
          { binding: 3, visibility: fragmentVisibility, sampler: { type: "filtering" } },
          { binding: 4, visibility: fragmentVisibility, buffer: { type: "uniform" } },
        ],
      });
      const presentBindGroupLayout = device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: fragmentVisibility,
            texture: { sampleType: "float", viewDimension: "2d" },
          },
          { binding: 1, visibility: fragmentVisibility, sampler: { type: "filtering" } },
        ],
      });
      const filterPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [filterBindGroupLayout],
      });
      const maskPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [maskBindGroupLayout],
      });
      const clutPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [clutBindGroupLayout],
      });
      const presentPipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [presentBindGroupLayout],
      });

      const filterPipeline = await createValidatedRenderPipeline(device, {
        layout: filterPipelineLayout,
        vertex: { module: filterModule, entryPoint: "vsMain" },
        fragment: {
          module: filterModule,
          entryPoint: "fsMain",
          targets: [{ format: "rgba8unorm" }],
        },
        primitive: { topology: "triangle-strip" },
      });

      const maskPipeline = await createValidatedRenderPipeline(device, {
        layout: maskPipelineLayout,
        vertex: { module: maskModule, entryPoint: "vsMain" },
        fragment: {
          module: maskModule,
          entryPoint: "fsMain",
          targets: [{ format: "rgba8unorm" }],
        },
        primitive: { topology: "triangle-strip" },
      });

      const clutPipeline = await createValidatedRenderPipeline(device, {
        layout: clutPipelineLayout,
        vertex: { module: clutModule, entryPoint: "vsMain" },
        fragment: {
          module: clutModule,
          entryPoint: "fsMain",
          targets: [{ format: "rgba8unorm" }],
        },
        primitive: { topology: "triangle-strip" },
      });

      const presentPipeline = await createValidatedRenderPipeline(device, {
        layout: presentPipelineLayout,
        vertex: { module: presentModule, entryPoint: "vsMain" },
        fragment: {
          module: presentModule,
          entryPoint: "fsMain",
          targets: [{ format: canvasFormat }],
        },
        primitive: { topology: "triangle-strip" },
      });

      return new WebGPUPreviewPipeline(
        canvas,
        device,
        context,
        canvasFormat,
        filterPipeline,
        maskPipeline,
        clutPipeline,
        presentPipeline,
        filterBindGroupLayout,
        maskBindGroupLayout,
        clutBindGroupLayout,
        presentBindGroupLayout,
      );
    } catch {
      return null;
    }
  }

  private ensureWorkingTextures(width: number, height: number) {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    if (this.pingTextureA && this.pingTextureB && this.pingWidth === w && this.pingHeight === h) {
      return;
    }
    this.pingTextureA?.destroy();
    this.pingTextureB?.destroy();
    this.pingTextureA = this.device.createTexture({
      size: { width: w, height: h },
      format: this.workingFormat,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });
    this.pingTextureB = this.device.createTexture({
      size: { width: w, height: h },
      format: this.workingFormat,
      usage:
        GPUTextureUsage.RENDER_ATTACHMENT |
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST,
    });
    this.pingWidth = w;
    this.pingHeight = h;
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.context.configure({
        device: this.device,
        format: this.canvasFormat,
        alphaMode: "premultiplied",
      });
    }
  }

  private beginPass(encoder: GPUCommandEncoder, targetView: GPUTextureView): GPURenderPassEncoder {
    return encoder.beginRenderPass({
      colorAttachments: [
        {
          view: targetView,
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          storeOp: "store",
        },
      ],
    });
  }

  private updateFilterUniforms(width: number, height: number, params: FilterParams, timestampSeconds: number) {
    const tint = parseHexColor(params.colorTintColor, [1, 1, 1]);
    const data = new Float32Array(24);
    data[0] = width;
    data[1] = height;
    data[2] = (params.brightness ?? 0) / 100;
    data[3] = 1 + (params.contrast ?? 0) / 100;

    data[4] = (params.hue ?? 0) / 100;
    data[5] = 1 + (params.saturation ?? 0) / 100;
    data[6] = clamp01((params.blur ?? 0) / 100);
    data[7] = clamp01((params.sharpness ?? 0) / 100);

    data[8] = clamp01((params.noise ?? 0) / 100);
    data[9] = clamp01((params.vignette ?? 0) / 100);
    data[10] = clamp01((params.colorTintIntensity ?? 0) / 100);
    data[11] = clamp01((params.scanLines ?? 0) / 100);

    data[12] = clamp01((params.chromaticAberration ?? 0) / 100);
    data[13] = clamp01((params.interlace ?? 0) / 100);
    data[14] = clamp01((params.pixelate ?? 0) / 100);
    data[15] = clamp01((params.jitter ?? 0) / 100);

    data[16] = tint[0];
    data[17] = tint[1];
    data[18] = tint[2];
    data[19] = timestampSeconds;
    this.device.queue.writeBuffer(this.filterUniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }

  private updateMaskUniforms(width: number, height: number, mask: NormalizedMaskPass) {
    const data = new Float32Array(24);
    data[0] = width;
    data[1] = height;
    data[2] = mask.toolKind;
    data[3] = mask.inverted;

    data[4] = mask.backgroundEnabled;
    data[5] = mask.maskEnabled;
    data[6] = mask.backgroundOpacity;
    data[7] = mask.maskOpacity;

    data[8] = mask.backgroundColor[0];
    data[9] = mask.backgroundColor[1];
    data[10] = mask.backgroundColor[2];
    data[11] = mask.shapeKind;

    data[12] = mask.maskColor[0];
    data[13] = mask.maskColor[1];
    data[14] = mask.maskColor[2];
    data[15] = mask.contours.length / 2;

    data[16] = mask.shapeBounds[0];
    data[17] = mask.shapeBounds[1];
    data[18] = mask.shapeBounds[2];
    data[19] = mask.shapeBounds[3];

    data[20] = mask.shapeScale[0];
    data[21] = mask.shapeScale[1];
    data[22] = mask.rotation;
    data[23] = mask.points.length / 2;
    this.device.queue.writeBuffer(this.maskUniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }

  private updateClutUniforms(clut: ClutTextureInfo, strength: number) {
    const data = new Float32Array([clut.size, clut.width, clamp01(strength), 0]);
    this.device.queue.writeBuffer(this.clutUniformBuffer, 0, data.buffer, data.byteOffset, data.byteLength);
  }

  private buildStorageBufferFromF32(values: Float32Array): GPUBuffer {
    if (values.length === 0) return this.dummyPointsBuffer;
    const byteLength = Math.max(8, values.byteLength);
    const buffer = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, values.buffer, values.byteOffset, values.byteLength);
    return buffer;
  }

  private buildStorageBufferFromU32(values: Uint32Array): GPUBuffer {
    if (values.length === 0) return this.dummyContoursBuffer;
    const byteLength = Math.max(8, values.byteLength);
    const buffer = this.device.createBuffer({
      size: byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(buffer, 0, values.buffer, values.byteOffset, values.byteLength);
    return buffer;
  }

  private async loadClutTexture(path: string): Promise<void> {
    if (this.clutCache.has(path) || this.clutLoading.has(path)) return;
    const promise = (async () => {
      const buffer = await readFileBuffer(path);
      const blob = new Blob([new Uint8Array(buffer)]);
      const bitmap = await createImageBitmap(blob);
      const texture = this.device.createTexture({
        size: { width: bitmap.width, height: bitmap.height },
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.device.queue.copyExternalImageToTexture(
        { source: bitmap },
        { texture },
        { width: bitmap.width, height: bitmap.height },
      );
      bitmap.close();

      const width = bitmap.width;
      const height = bitmap.height;
      const length = Math.min(width, height);
      let level = 2;
      while (level * level * level < length) {
        level++;
      }
      level = level * level;
      this.clutCache.set(path, {
        texture,
        width,
        size: level,
      });
    })().finally(() => {
      this.clutLoading.delete(path);
    });
    this.clutLoading.set(path, promise);
    await promise;
  }

  warmupFilterApplicators(applicators: BaseClipApplicator[]): void {
    const paths = extractAllFilterApplicatorPaths(applicators);
    for (const path of paths) {
      if (!this.clutCache.has(path) && !this.clutLoading.has(path)) {
        void this.loadClutTexture(path);
      }
    }
  }

  render(params: {
    source: VideoFrame | HTMLCanvasElement | OffscreenCanvas;
    width: number;
    height: number;
    filterParams: FilterParams;
    masks: MaskClipProps[];
    maskFrame: number;
    clipTransform?: ClipTransform;
    originalClipTransform?: ClipTransform;
    applicators: BaseClipApplicator[];
    focusFrame: number;
    timestampSeconds: number;
  }): boolean {
    if (!this.pingTextureA || !this.pingTextureB || this.pingWidth !== params.width || this.pingHeight !== params.height) {
      this.ensureWorkingTextures(params.width, params.height);
    }
    if (!this.pingTextureA || !this.pingTextureB) return false;

    this.updateFilterUniforms(params.width, params.height, params.filterParams, params.timestampSeconds);

    const encoder = this.device.createCommandEncoder();

    // Pass 1: source -> ping A with full filter stack.
    const externalTexture = this.device.importExternalTexture({
      source: params.source as any,
    });
    const filterBindGroup = this.device.createBindGroup({
      layout: this.filterBindGroupLayout,
      entries: [
        { binding: 0, resource: externalTexture },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: { buffer: this.filterUniformBuffer } },
      ],
    });

    {
      const pass = this.beginPass(encoder, this.pingTextureA.createView());
      pass.setPipeline(this.filterPipeline);
      pass.setBindGroup(0, filterBindGroup);
      pass.draw(4, 1, 0, 0);
      pass.end();
    }

    let srcTex = this.pingTextureA;
    let dstTex = this.pingTextureB;

    // Pass 2+: masks
    for (let i = 0; i < params.masks.length; i++) {
      const normalized = normalizeMaskPass(
        params.masks[i],
        params.maskFrame,
        params.width,
        params.height,
        params.clipTransform,
        params.originalClipTransform,
        i === 0,
      );
      if (!normalized) continue;
      this.updateMaskUniforms(params.width, params.height, normalized);
      const pointsBuffer = this.buildStorageBufferFromF32(normalized.points);
      const contoursBuffer = this.buildStorageBufferFromU32(normalized.contours);
      const bindGroup = this.device.createBindGroup({
        layout: this.maskBindGroupLayout,
        entries: [
          { binding: 0, resource: srcTex.createView() },
          { binding: 1, resource: this.linearSampler },
          { binding: 2, resource: { buffer: this.maskUniformBuffer } },
          { binding: 3, resource: { buffer: pointsBuffer } },
          { binding: 4, resource: { buffer: contoursBuffer } },
        ],
      });

      const pass = this.beginPass(encoder, dstTex.createView());
      pass.setPipeline(this.maskPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(4, 1, 0, 0);
      pass.end();

      if (pointsBuffer !== this.dummyPointsBuffer) pointsBuffer.destroy();
      if (contoursBuffer !== this.dummyContoursBuffer) contoursBuffer.destroy();
      const tmp = srcTex;
      srcTex = dstTex;
      dstTex = tmp;
    }

    // Pass N+: filter applicators (CLUT)
    const activeFilterApplicators = extractActiveFilterApplicators(params.applicators, params.focusFrame);
    for (const applicator of activeFilterApplicators) {
      const clut = this.clutCache.get(applicator.path);
      if (!clut) {
        if (!this.clutLoading.has(applicator.path)) {
          void this.loadClutTexture(applicator.path);
        }
        continue;
      }

      this.updateClutUniforms(clut, applicator.strength);
      const bindGroup = this.device.createBindGroup({
        layout: this.clutBindGroupLayout,
        entries: [
          { binding: 0, resource: srcTex.createView() },
          { binding: 1, resource: this.linearSampler },
          { binding: 2, resource: clut.texture.createView() },
          { binding: 3, resource: this.nearestSampler },
          { binding: 4, resource: { buffer: this.clutUniformBuffer } },
        ],
      });

      const pass = this.beginPass(encoder, dstTex.createView());
      pass.setPipeline(this.clutPipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(4, 1, 0, 0);
      pass.end();

      const tmp = srcTex;
      srcTex = dstTex;
      dstTex = tmp;
    }

    // Final pass: working texture -> canvas
    const presentBindGroup = this.device.createBindGroup({
      layout: this.presentBindGroupLayout,
      entries: [
        { binding: 0, resource: srcTex.createView() },
        { binding: 1, resource: this.linearSampler },
      ],
    });
    {
      const pass = this.beginPass(encoder, this.context.getCurrentTexture().createView());
      pass.setPipeline(this.presentPipeline);
      pass.setBindGroup(0, presentBindGroup);
      pass.draw(4, 1, 0, 0);
      pass.end();
    }

    this.device.queue.submit([encoder.finish()]);
    return true;
  }

  dispose() {
    this.pingTextureA?.destroy();
    this.pingTextureB?.destroy();
    this.pingTextureA = null;
    this.pingTextureB = null;
    this.filterUniformBuffer.destroy();
    this.maskUniformBuffer.destroy();
    this.clutUniformBuffer.destroy();
    this.dummyPointsBuffer.destroy();
    this.dummyContoursBuffer.destroy();
    for (const clut of this.clutCache.values()) {
      clut.texture.destroy();
    }
    this.clutCache.clear();
    this.clutLoading.clear();
  }
}
