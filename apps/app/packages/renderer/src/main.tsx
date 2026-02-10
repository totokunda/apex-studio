import React from "react";
import ReactDOM from "react-dom/client";
import App from "./components/App";
import Launcher from "./components/Launcher";
import { VideoDecoderManagerProvider } from "@/lib/media/VideoDecoderManagerContext";

import "./styles/index.css";
import "./fonts";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <VideoDecoderManagerProvider>
      {window.location.hash === "#launcher" ? <Launcher /> : <App />}
    </VideoDecoderManagerProvider>
  </React.StrictMode>,
);
