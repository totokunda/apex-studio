import { load, stop } from "../src/index.ts";

const reader = load("/Users/tosinkuye/Downloads/11933881_2160_3840_30fps.mp4", 4);

console.log(`Resolution: ${reader.width}x${reader.height}`);
console.log(`YUV420p frame size: ${reader.frameSize} bytes`);
console.log(`Slots: ${reader.slotCount}`);

let count = 0;
const start = performance.now();

// Poll for frames — in a real app you'd do this in a requestAnimationFrame
// loop or a worker with Atomics.wait()
const interval = setInterval(() => {
  // Drain all available frames per tick
  let frame;
  while ((frame = reader.readFrame()) !== null) {
    count++;
    // `frame` is a zero-copy Uint8Array view into shared memory.
    // YUV420p layout:
    //   Y plane: frame.subarray(0, width * height)
    //   U plane: frame.subarray(width * height, width * height * 5/4)
    //   V plane: frame.subarray(width * height * 5/4, width * height * 3/2)
  }
}, 1); // 1ms poll interval

setTimeout(() => {
  clearInterval(interval);
  stop();
  const elapsed = (performance.now() - start) / 1000;
  console.log(`\nProcessed ${count} frames in ${elapsed.toFixed(2)}s`);
  console.log(`Throughput: ${(count / elapsed).toFixed(1)} fps`);
  console.log(`Dropped: ${reader.droppedFrames()} frames`);
}, 20000);