export const generateTimelineThumbnailShape = async (
  clipType: string,
  imageCanvas: HTMLCanvasElement,
  groupRef: any,
) => {
  if (clipType !== "shape") return;
  // make canvas
  const ctx = imageCanvas.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.fillStyle = "#894c30";
    ctx.fillRect(0, 0, imageCanvas.width, imageCanvas.height);
  }
  groupRef.current?.getLayer()?.batchDraw();
};
