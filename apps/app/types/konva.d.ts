import KonvaLocal from "konva";

declare namespace Konva {
  export const Filters: {
    Applicator: (imageData: ImageData) => void;
  } & typeof KonvaLocal.Filters;
}
