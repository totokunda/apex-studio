import { AppModule } from "../AppModule.js";
import { ModuleContext } from "../ModuleContext.js";

export class HardwareAccelerationModule implements AppModule {
  readonly #shouldBeDisabled: boolean;

  constructor({ enable }: { enable: boolean }) {
    this.#shouldBeDisabled = !enable;
  }

  enable({ app }: ModuleContext): Promise<void> | void {
    if (this.#shouldBeDisabled) {
      app.disableHardwareAcceleration();
      app.getGPUInfo("complete").then(console.log);
    }

    if (process.platform === "win32") {
      // Force Chromium to use the D3D11 ANGLE backend instead of the D3D9
      // fallback. D3D11 has a lower-latency video decode pipeline (D3D11VA)
      // that dramatically reduces per-frame decode time for WebCodecs on
      // Windows — the primary cause of playback jitter vs macOS VideoToolbox.
      app.commandLine.appendSwitch("use-angle", "d3d11");

      // Enable the D3D11 VideoDecoder feature flag so WebCodecs can access
      // hardware-accelerated D3D11VA decode surfaces directly.
      app.commandLine.appendSwitch("enable-features", "D3D11VideoDecoder,PlatformHEVCDecoderSupport");

      // Disable D3D11 debug layers in production to avoid GPU validation
      // overhead that adds latency on some Windows driver versions.
      app.commandLine.appendSwitch("disable-features", "D3D11DebugLayer");

      // Allow higher GPU memory limits so video frame buffers aren't evicted
      // during decode, which causes re-decodes and visible frame drops.
      app.commandLine.appendSwitch("gpu-memory-buffer-compositor-resources");
    }
  }


}

export function hardwareAccelerationMode(
  ...args: ConstructorParameters<typeof HardwareAccelerationModule>
) {
  return new HardwareAccelerationModule(...args);
}
