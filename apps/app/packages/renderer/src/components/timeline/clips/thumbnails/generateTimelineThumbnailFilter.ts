export const generateTimelineThumbnailFilter = async (
  clipType: string,
  imageCanvas: HTMLCanvasElement,
  groupRef: any,
) => {
  if (clipType !== "filter") return;
  // make canvas
  const ctx = imageCanvas.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.fillStyle = "#02ace6";
    ctx.fillRect(0, 0, imageCanvas.width, imageCanvas.height);
  }
  groupRef.current?.getLayer()?.batchDraw();
};
