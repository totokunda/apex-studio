/**
 * Combined WebGL Fragment Shader with #ifdef blocks for all effects.
 *
 * Define any of these before compiling to include that effect:
 *   APPLY_BRIGHTNESS
 *   APPLY_CHROMATIC_ABERRATION
 *   APPLY_COLOR_TINT
 *   APPLY_CONTRAST
 *   APPLY_HUE_SATURATION
 *   APPLY_INTERLACE
 *   APPLY_JITTER
 *   APPLY_NOISE
 *   APPLY_PIXELATE
 *   APPLY_SCAN_LINES
 *   APPLY_SHARPNESS
 *   APPLY_VIGNETTE
 *
 * Required uniforms (define-dependent):
 *   u_image (sampler2D) - always
 *   u_resolution (vec2) - for pixelate, scan-lines, interlace, sharpness
 *   u_brightness (float) - for brightness
 *   u_contrast (float) - for contrast
 *   u_hue (float), u_saturation (float) - for hue-saturation
 *   u_tintColor (vec3), u_tintIntensity (float) - for color-tint
 *   u_amount (float) - for chromatic aberration
 *   u_jitterIntensity (float), u_offset (vec2) - for jitter
 *   u_pixelSize (float) - for pixelate
 *   u_noise (float), u_seed (float) - for noise
 *   u_interlaceIntensity (float), u_interlaceSeed (float) - for interlace
 *   u_scanLinesIntensity (float) - for scan-lines
 *   u_sharpness (float) - for sharpness
 *   u_vignette (float) - for vignette
 */

precision mediump float;

uniform sampler2D u_image;
uniform vec2 u_resolution;

#ifdef APPLY_BRIGHTNESS
uniform float u_brightness;
#endif

#ifdef APPLY_CONTRAST
uniform float u_contrast;
#endif

#ifdef APPLY_HUE_SATURATION
uniform float u_hue;
uniform float u_saturation;
#endif

#ifdef APPLY_COLOR_TINT
uniform vec3 u_tintColor;
uniform float u_tintIntensity;
#endif

#ifdef APPLY_CHROMATIC_ABERRATION
uniform float u_amount;
#endif

#ifdef APPLY_JITTER
uniform float u_jitterIntensity;
uniform vec2 u_offset;
#endif

#ifdef APPLY_PIXELATE
uniform float u_pixelSize;
#endif

#ifdef APPLY_NOISE
uniform float u_noise;
uniform float u_seed;
#endif

#ifdef APPLY_INTERLACE
uniform float u_interlaceIntensity;
uniform float u_interlaceSeed;
#endif

#ifdef APPLY_SCAN_LINES
uniform float u_scanLinesIntensity;
#endif

#ifdef APPLY_SHARPNESS
uniform float u_sharpness;
#endif

#ifdef APPLY_VIGNETTE
uniform float u_vignette;
#endif

varying vec2 v_texCoord;

#ifdef APPLY_HUE_SATURATION
// RGB to HSL conversion
vec3 rgb2hsl(vec3 color) {
  float maxC = max(max(color.r, color.g), color.b);
  float minC = min(min(color.r, color.g), color.b);
  float delta = maxC - minC;

  float h = 0.0;
  float s = 0.0;
  float l = (maxC + minC) / 2.0;

  if (delta > 0.0001) {
    s = l < 0.5 ? delta / (maxC + minC) : delta / (2.0 - maxC - minC);

    if (color.r >= maxC) {
      h = (color.g - color.b) / delta + (color.g < color.b ? 6.0 : 0.0);
    } else if (color.g >= maxC) {
      h = (color.b - color.r) / delta + 2.0;
    } else {
      h = (color.r - color.g) / delta + 4.0;
    }
    h /= 6.0;
  }

  return vec3(h, s, l);
}

float hue2rgb(float p, float q, float t) {
  if (t < 0.0) t += 1.0;
  if (t > 1.0) t -= 1.0;
  if (t < 1.0/6.0) return p + (q - p) * 6.0 * t;
  if (t < 1.0/2.0) return q;
  if (t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0;
  return p;
}

vec3 hsl2rgb(vec3 hsl) {
  float h = hsl.x;
  float s = hsl.y;
  float l = hsl.z;

  if (s == 0.0) {
    return vec3(l, l, l);
  }

  float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s;
  float p = 2.0 * l - q;

  return vec3(
    hue2rgb(p, q, h + 1.0/3.0),
    hue2rgb(p, q, h),
    hue2rgb(p, q, h - 1.0/3.0)
  );
}
#endif

#ifdef APPLY_NOISE
float random(vec2 co) {
  return fract(sin(dot(co.xy, vec2(12.9898, 78.233)) + u_seed) * 43758.5453);
}
#endif

#ifdef APPLY_INTERLACE
float interlaceRandom(float x) {
  return fract(sin(x * 12.9898 + u_interlaceSeed) * 43758.5453);
}
#endif

void main() {
  vec2 uv = v_texCoord;

  #ifdef APPLY_JITTER
  uv = uv + u_offset * u_jitterIntensity;
  uv = clamp(uv, vec2(0.0), vec2(1.0));
  #endif

  #ifdef APPLY_PIXELATE
  vec2 blockSize = u_pixelSize / u_resolution;
  vec2 blockCoord = floor(uv / blockSize) * blockSize + blockSize * 0.5;
  uv = clamp(blockCoord, vec2(0.0), vec2(1.0));
  #endif

  vec4 color;

  #ifdef APPLY_CHROMATIC_ABERRATION
  vec2 center = vec2(0.5, 0.5);
  vec2 dir = uv - center;
  float chromaDist = length(dir);
  float offset = u_amount * 0.015 * chromaDist;

  float r = texture2D(u_image, uv + dir * offset).r;
  float g = texture2D(u_image, uv).g;
  float b = texture2D(u_image, uv - dir * offset).b;
  float a = texture2D(u_image, uv).a;
  color = vec4(r, g, b, a);
  #else
  color = texture2D(u_image, uv);
  #endif

  #ifdef APPLY_INTERLACE
  float interlaceY = gl_FragCoord.y;
  bool isOddLine = mod(interlaceY, 2.0) < 1.0;
  vec2 interlaceUv = uv;
  if (isOddLine) {
    float fieldShift = u_interlaceIntensity * 0.003 * (interlaceRandom(floor(interlaceY)) - 0.5);
    interlaceUv.x += fieldShift;
  }
  vec4 interlaceColor = texture2D(u_image, clamp(interlaceUv, vec2(0.0), vec2(1.0)));
  float interlaceDimFactor = isOddLine ? (1.0 - u_interlaceIntensity * 0.15) : 1.0;
  vec2 offsetUV = interlaceUv + vec2(0.0, 1.0 / u_resolution.y);
  vec4 neighborColor = texture2D(u_image, clamp(offsetUV, vec2(0.0), vec2(1.0)));
  float blendAmount = isOddLine ? u_interlaceIntensity * 0.2 : 0.0;
  color = mix(interlaceColor, neighborColor, blendAmount);
  color.rgb *= interlaceDimFactor;
  #endif

  #ifdef APPLY_BRIGHTNESS
  color.rgb += u_brightness;
  #endif

  #ifdef APPLY_CONTRAST
  float factor = (1.0 + u_contrast);
  color.rgb = (color.rgb - 0.5) * factor + 0.5;
  #endif

  #ifdef APPLY_HUE_SATURATION
  vec3 hsl = rgb2hsl(color.rgb);
  hsl.x = mod(hsl.x + u_hue, 1.0);
  hsl.y = clamp(hsl.y * (1.0 + u_saturation), 0.0, 1.0);
  color.rgb = hsl2rgb(hsl);
  #endif

  #ifdef APPLY_COLOR_TINT
  float luminance = dot(color.rgb, vec3(0.299, 0.587, 0.114));
  vec3 tinted = luminance * u_tintColor;
  color.rgb = mix(color.rgb, tinted, u_tintIntensity);
  #endif

  #ifdef APPLY_SHARPNESS
  vec2 texel = 1.0 / u_resolution;
  vec4 centerSample = color;
  vec4 top = texture2D(u_image, uv + vec2(0.0, -texel.y));
  vec4 bottom = texture2D(u_image, uv + vec2(0.0, texel.y));
  vec4 left = texture2D(u_image, uv + vec2(-texel.x, 0.0));
  vec4 right = texture2D(u_image, uv + vec2(texel.x, 0.0));
  vec4 topLeft = texture2D(u_image, uv + vec2(-texel.x, -texel.y));
  vec4 topRight = texture2D(u_image, uv + vec2(texel.x, -texel.y));
  vec4 bottomLeft = texture2D(u_image, uv + vec2(-texel.x, texel.y));
  vec4 bottomRight = texture2D(u_image, uv + vec2(texel.x, texel.y));

  vec4 blurred = (
    topLeft + top * 2.0 + topRight +
    left * 2.0 + centerSample * 4.0 + right * 2.0 +
    bottomLeft + bottom * 2.0 + bottomRight
  ) / 16.0;

  float strength = u_sharpness * 5.0;
  vec4 sharpened = centerSample + strength * (centerSample - blurred);
  color = vec4(sharpened.rgb, centerSample.a);
  #endif

  #ifdef APPLY_NOISE
  float noiseVal = (random(uv) - 0.5) * u_noise;
  color.rgb += noiseVal;
  #endif

  #ifdef APPLY_SCAN_LINES
  float scanLineY = gl_FragCoord.y;
  float lineSpacing = 3.0;
  float scanLine = 0.5 + 0.5 * sin(scanLineY * 3.14159265 / (lineSpacing * 0.5));
  float scanDimFactor = mix(1.0, scanLine, u_scanLinesIntensity * 0.6);
  float largeBand = 0.5 + 0.5 * sin(scanLineY * 3.14159265 / (u_resolution.y * 0.25));
  float bandFactor = mix(1.0, 0.92 + 0.08 * largeBand, u_scanLinesIntensity);
  color.rgb *= scanDimFactor * bandFactor;
  #endif

  #ifdef APPLY_VIGNETTE
  vec2 diff = v_texCoord - vec2(0.5, 0.5);
  float vignetteDist = length(diff);
  float maxDist = length(vec2(0.5, 0.5));
  float normDist = vignetteDist / maxDist;
  float falloff = 1.0 - (pow(normDist, 3.0) * u_vignette * 2.0);
  falloff = clamp(falloff, 0.0, 1.0);
  color.rgb *= falloff;
  #endif

  gl_FragColor = vec4(color.rgb, color.a);
}
