/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'battle-dark': '#0f172a',
        'battle-blue': '#3b82f6',
        'battle-red': '#ef4444',
        'battle-gold': '#fbbf24',
      },
      animation: {
        'slide-in-left': 'slideInLeft 1s ease-out forwards',
        'slide-in-right': 'slideInRight 1s ease-out forwards',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'battle-shake': 'battleShake 0.5s ease-in-out infinite',
        'spark': 'spark 0.5s ease-out forwards',
        'winner-glow': 'winnerGlow 2s ease-in-out infinite',
      },
      keyframes: {
        slideInLeft: {
          '0%': { transform: 'translateX(-200%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        slideInRight: {
          '0%': { transform: 'translateX(200%)', opacity: '0' },
          '100%': { transform: 'translateX(0)', opacity: '1' },
        },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 20px rgba(59, 130, 246, 0.5)' },
          '50%': { boxShadow: '0 0 40px rgba(59, 130, 246, 0.8)' },
        },
        battleShake: {
          '0%, 100%': { transform: 'translateX(0)' },
          '25%': { transform: 'translateX(-5px)' },
          '75%': { transform: 'translateX(5px)' },
        },
        spark: {
          '0%': { transform: 'scale(0)', opacity: '1' },
          '100%': { transform: 'scale(2)', opacity: '0' },
        },
        winnerGlow: {
          '0%, 100%': { boxShadow: '0 0 30px rgba(251, 191, 36, 0.6)' },
          '50%': { boxShadow: '0 0 60px rgba(251, 191, 36, 1)' },
        },
      },
    },
  },
  plugins: [],
}
