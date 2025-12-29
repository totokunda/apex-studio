export const generateTimelineThumbnailDrawing = async (
  clipType: string,
  imageCanvas: HTMLCanvasElement,
  clipRef: any,
) => {
  if (clipType !== "draw") return;
  // make canvas
  const ctx = imageCanvas.getContext("2d");
  if (ctx) {
    ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
    ctx.fillStyle = "#9B59B6";
    ctx.fillRect(0, 0, imageCanvas.width, imageCanvas.height);
  }
  clipRef.current?.getLayer()?.batchDraw();
};
