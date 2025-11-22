/**********************
 Tailwind CSS Configuration
**********************/
/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: 'class',
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: '#faf5ff',
          100: '#f3e8ff',
          200: '#e9d5ff',
          300: '#d8b4fe',
          400: '#c084fc',
          500: '#a855f7',
          600: '#9333ea',
          700: '#7e22ce',
          800: '#6b21a8',
          900: '#581c87',
        },
        surface: {
          900: '#0b0b0d', // pitch-black base
          800: '#0f0f13',
          700: '#14141a',
        }
      },
      boxShadow: {
        card: '0 8px 32px rgba(0,0,0,0.35)',
        glow: '0 0 24px rgba(168,85,247,0.35)',
      },
    },
  },
  plugins: [],
}
