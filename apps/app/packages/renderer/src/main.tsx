import React from "react";
import ReactDOM from "react-dom/client";
import App from "./components/App";
import Launcher from "./components/Launcher";

import "./styles/index.css";

// Import Google Fonts
import "@fontsource/poppins/400.css";
import "@fontsource/poppins/400-italic.css";
import "@fontsource/poppins/500.css";
import "@fontsource/poppins/500-italic.css";
import "@fontsource/poppins/600.css";
import "@fontsource/poppins/600-italic.css";
import "@fontsource/poppins/700.css";
import "@fontsource/poppins/700-italic.css";

import "@fontsource/montserrat/400.css";
import "@fontsource/montserrat/400-italic.css";
import "@fontsource/montserrat/500.css";
import "@fontsource/montserrat/500-italic.css";
import "@fontsource/montserrat/600.css";
import "@fontsource/montserrat/600-italic.css";
import "@fontsource/montserrat/700.css";
import "@fontsource/montserrat/700-italic.css";

import "@fontsource/roboto/400.css";
import "@fontsource/roboto/400-italic.css";
import "@fontsource/roboto/500.css";
import "@fontsource/roboto/500-italic.css";
import "@fontsource/roboto/600.css";
import "@fontsource/roboto/600-italic.css";
import "@fontsource/roboto/700.css";
import "@fontsource/roboto/700-italic.css";

import "@fontsource/open-sans/400.css";
import "@fontsource/open-sans/400-italic.css";
import "@fontsource/open-sans/500.css";
import "@fontsource/open-sans/500-italic.css";
import "@fontsource/open-sans/600.css";
import "@fontsource/open-sans/600-italic.css";
import "@fontsource/open-sans/700.css";
import "@fontsource/open-sans/700-italic.css";

import "@fontsource/lato/400.css";
import "@fontsource/lato/400-italic.css";
import "@fontsource/lato/700.css";
import "@fontsource/lato/700-italic.css";

import "@fontsource/oswald/400.css";
import "@fontsource/oswald/500.css";
import "@fontsource/oswald/600.css";
import "@fontsource/oswald/700.css";

import "@fontsource/raleway/400.css";
import "@fontsource/raleway/400-italic.css";
import "@fontsource/raleway/500.css";
import "@fontsource/raleway/500-italic.css";
import "@fontsource/raleway/600.css";
import "@fontsource/raleway/600-italic.css";
import "@fontsource/raleway/700.css";
import "@fontsource/raleway/700-italic.css";

import "@fontsource/pt-sans/400.css";
import "@fontsource/pt-sans/400-italic.css";
import "@fontsource/pt-sans/700.css";
import "@fontsource/pt-sans/700-italic.css";

import "@fontsource/merriweather/400.css";
import "@fontsource/merriweather/400-italic.css";
import "@fontsource/merriweather/500.css";
import "@fontsource/merriweather/500-italic.css";
import "@fontsource/merriweather/600.css";
import "@fontsource/merriweather/600-italic.css";
import "@fontsource/merriweather/700.css";
import "@fontsource/merriweather/700-italic.css";

import "@fontsource/playfair-display/400.css";
import "@fontsource/playfair-display/400-italic.css";
import "@fontsource/playfair-display/500.css";
import "@fontsource/playfair-display/500-italic.css";
import "@fontsource/playfair-display/600.css";
import "@fontsource/playfair-display/600-italic.css";
import "@fontsource/playfair-display/700.css";
import "@fontsource/playfair-display/700-italic.css";

import "@fontsource/nunito/400.css";
import "@fontsource/nunito/400-italic.css";
import "@fontsource/nunito/500.css";
import "@fontsource/nunito/500-italic.css";
import "@fontsource/nunito/600.css";
import "@fontsource/nunito/600-italic.css";
import "@fontsource/nunito/700.css";
import "@fontsource/nunito/700-italic.css";

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    {window.location.hash === "#launcher" ? <Launcher /> : <App />}
  </React.StrictMode>,
);
