/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        forge: {
          ember: "#ff6b35",
          flame: "#f7931e",
          steel: "#1e293b",
          iron: "#0f172a",
        },
      },
    },
  },
  plugins: [],
};
