/// <reference types="vite/client" />

declare module "*.glsl?raw" {
  const src: string;
  export default src;
}
