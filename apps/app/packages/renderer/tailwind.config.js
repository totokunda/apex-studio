export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        poppins: ["Poppins", "sans-serif"],
        montserrat: ["Montserrat", "sans-serif"],
        roboto: ["Roboto", "sans-serif"],
        "open-sans": ["Open Sans", "sans-serif"],
        lato: ["Lato", "sans-serif"],
        oswald: ["Oswald", "sans-serif"],
        raleway: ["Raleway", "sans-serif"],
        "tiktok-sans": ["TikTok Sans", "sans-serif"],
        "pt-sans": ["PT Sans", "sans-serif"],
        merriweather: ["Merriweather", "serif"],
        "playfair-display": ["Playfair Display", "serif"],
        nunito: ["Nunito", "sans-serif"],
        // Bold/Condensed Display Fonts
        "bebas-neue": ["Bebas Neue", "sans-serif"],
        anton: ["Anton", "sans-serif"],
        "archivo-black": ["Archivo Black", "sans-serif"],
        "barlow-condensed": ["Barlow Condensed", "sans-serif"],
        // Cinematic/Dramatic Serifs
        cinzel: ["Cinzel", "serif"],
        "abril-fatface": ["Abril Fatface", "serif"],
        "libre-baskerville": ["Libre Baskerville", "serif"],
        // Handwritten/Script Fonts
        "permanent-marker": ["Permanent Marker", "cursive"],
        bangers: ["Bangers", "cursive"],
        "dancing-script": ["Dancing Script", "cursive"],
        caveat: ["Caveat", "cursive"],
        // Modern Geometric Sans-Serifs
        inter: ["Inter", "sans-serif"],
        "dm-sans": ["DM Sans", "sans-serif"],
        "space-grotesk": ["Space Grotesk", "sans-serif"],
        // Monospace Fonts
        "source-code-pro": ["Source Code Pro", "monospace"],
        "jetbrains-mono": ["JetBrains Mono", "monospace"],
      },
      keyframes: {
        ripple: {
          "0%": {
            boxShadow: "0 0 0 0 rgba(255, 255, 255, 0.25)",
          },
          "70%": {
            boxShadow: "0 0 0 12px rgba(255, 255, 255, 0)",
          },
          "100%": {
            boxShadow: "0 0 0 0 rgba(255, 255, 255, 0)",
          },
        },
      },
      animation: {
        ripple: "ripple 1.6s ease-out infinite",
      },
      colors: {
        brand: {
          background: "#151517",
          DEFAULT: "#222124",
        },
      },
    },
  },
  plugins: [],
};
