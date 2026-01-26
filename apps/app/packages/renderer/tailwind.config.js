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
        "pt-sans": ["PT Sans", "sans-serif"],
        merriweather: ["Merriweather", "serif"],
        "playfair-display": ["Playfair Display", "serif"],
        nunito: ["Nunito", "sans-serif"],
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
