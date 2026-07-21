/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // AI Human Vision — ACUD-inspired red. `brand-600` is the primary
        // (#D62027); the rest is a tuned tint/shade ramp used across the UI.
        brand: {
          50: "#fdf0f0", 100: "#fbdadb", 200: "#f6bcbe", 300: "#f08f92",
          400: "#e85a5e", 500: "#e0343a", 600: "#d62027", 700: "#b71a20",
          800: "#971519", 900: "#7c1417", 950: "#430a0c",
        },
        success: "#2e7d32",
        warning: "#f9a825",
        danger: "#c62828",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      boxShadow: {
        // soft, premium elevation used by cards / navbar / dialogs
        soft: "0 1px 2px rgb(17 17 17 / 0.04), 0 6px 20px rgb(17 17 17 / 0.05)",
        "soft-lg": "0 2px 4px rgb(17 17 17 / 0.05), 0 12px 32px rgb(17 17 17 / 0.09)",
      },
      keyframes: {
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
        "slide-up": { from: { opacity: "0", transform: "translateY(10px)" }, to: { opacity: "1", transform: "translateY(0)" } },
        "slide-in-left": { from: { opacity: "0", transform: "translateX(-12px)" }, to: { opacity: "1", transform: "translateX(0)" } },
        "scale-in": { from: { opacity: "0", transform: "scale(0.96)" }, to: { opacity: "1", transform: "scale(1)" } },
        "pulse-dot": { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.35" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
        "logo-pulse": { "0%,100%": { transform: "scale(1)", opacity: "1" }, "50%": { transform: "scale(1.06)", opacity: "0.85" } },
      },
      animation: {
        "fade-in": "fade-in .28s ease-out",
        "slide-up": "slide-up .34s cubic-bezier(.16,1,.3,1)",
        "slide-in-left": "slide-in-left .3s cubic-bezier(.16,1,.3,1)",
        "scale-in": "scale-in .22s cubic-bezier(.16,1,.3,1)",
        "pulse-dot": "pulse-dot 1.4s ease-in-out infinite",
        "logo-pulse": "logo-pulse 1.8s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
