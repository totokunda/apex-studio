import { MediaInfo } from "../types";
import { Output, Mp4OutputFormat, BufferTarget, Conversion } from "mediabunny";

export const convertMediaTo24Fps = async (mediaInfo: MediaInfo) => {
  // Check if the media is already 24 fps

  const input = mediaInfo.originalInput;
  if (!input) throw new Error("No original input found");
  const output = new Output({
    format: new Mp4OutputFormat(),
    target: new BufferTarget(),
  });
  const conversion = await Conversion.init({
    input: input,
    output,
    video: {
      frameRate: 24,
    },
  });
  await conversion.execute();
  const buffer = await output.target.buffer;
  return buffer;
};
