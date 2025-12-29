import Konva from "konva";
import { BaseClipApplicator } from "../apply/base";

interface ApplicatorFilterNode extends Konva.Node {
  applicators?: BaseClipApplicator[];
}

function ApplicatorFilter(this: ApplicatorFilterNode, imageData: ImageData) {
  //@ts-ignore
  const applicators = this.attrs.applicators as BaseClipApplicator[];
  if (!applicators || applicators.length === 0) {
    return;
  }

  // convert imageData to canvas
  let canvas = document.createElement("canvas");
  canvas.width = imageData.width;
  canvas.height = imageData.height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.putImageData(imageData, 0, 0);

  applicators.forEach((applicator) => {
    canvas = applicator.apply(canvas);
  });

  // convert canvas back to imageData and copy to original imageData
  const newImageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  imageData.data.set(newImageData.data);
}

export default ApplicatorFilter;
