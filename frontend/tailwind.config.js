/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
      colors: { accent: { DEFAULT: "#e60023", soft: "#fde8eb" } },
    },
  },
  plugins: [],
};
