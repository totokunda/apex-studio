export default {
    content: ["./index.html","./src/**/*.{js,ts,jsx,tsx}"],
    theme: { 
    extend: {
        fontFamily: {
            poppins: ['Poppins', 'sans-serif'],
          },
          colors: {
            brand: {
                background: '#151517',
                DEFAULT: '#222124',
            }
          },
    } },
    plugins: [],
}