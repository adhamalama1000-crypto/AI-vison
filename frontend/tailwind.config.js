/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Madkour AI Panel Inspector — engineering blue. `brand-600` is the
        // primary (#2D8CDC); the ramp is used for tints, borders and states.
        brand: {
          50: "#eef7fe", 100: "#d6ecfc", 200: "#addaf9", 300: "#7cc2f2",
          400: "#4ea6e6", 500: "#2d8cdc", 600: "#1f72bd", 700: "#1a5c9a",
          800: "#184c7c", 900: "#173f66", 950: "#0f2841",
        },
        // signal amber — advisory findings, DIN-rail accents, gauges
        signal: {
          50: "#fdf6e7", 100: "#faeac1", 200: "#f6d888", 300: "#f2c453",
          400: "#f0ad2e", 500: "#dc9317", 600: "#b87312", 700: "#935a13",
          800: "#7a4a16", 900: "#673e16", 950: "#3c2108",
        },
        success: "#3dc78a",
        warning: "#f0ad2e",
        danger: "#ec5656",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica", "Arial", "sans-serif"],
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
      },
      boxShadow: {
        // machined elevation: a crisp top edge plus a deep, diffuse drop, which
        // reads as instrument panelling rather than as a floating consumer card
        soft: "inset 0 1px 0 rgb(255 255 255 / 0.04), 0 1px 2px rgb(0 0 0 / 0.35), 0 8px 24px rgb(0 0 0 / 0.22)",
        "soft-lg": "inset 0 1px 0 rgb(255 255 255 / 0.05), 0 2px 6px rgb(0 0 0 / 0.4), 0 18px 44px rgb(0 0 0 / 0.32)",
      },
      backgroundImage: {
        // faint DIN-rail hatch used behind hero panels
        "din-rail": "repeating-linear-gradient(90deg, rgb(255 255 255 / 0.045) 0 1px, transparent 1px 14px)",
      },
      keyframes: {
        "fade-in": { from: { opacity: "0" }, to: { opacity: "1" } },
        "slide-up": { from: { opacity: "0", transform: "translateY(10px)" }, to: { opacity: "1", transform: "translateY(0)" } },
        "slide-in-left": { from: { opacity: "0", transform: "translateX(-12px)" }, to: { opacity: "1", transform: "translateX(0)" } },
        "scale-in": { from: { opacity: "0", transform: "scale(0.96)" }, to: { opacity: "1", transform: "scale(1)" } },
        "pulse-dot": { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.35" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
        "logo-pulse": { "0%,100%": { transform: "scale(1)", opacity: "1" }, "50%": { transform: "scale(1.06)", opacity: "0.85" } },
        "sweep": { from: { transform: "translateY(-100%)" }, to: { transform: "translateY(400%)" } },
        "count-up": { from: { opacity: "0", transform: "translateY(6px)" }, to: { opacity: "1", transform: "translateY(0)" } },
      },
      animation: {
        "fade-in": "fade-in .28s ease-out",
        "slide-up": "slide-up .34s cubic-bezier(.16,1,.3,1)",
        "slide-in-left": "slide-in-left .3s cubic-bezier(.16,1,.3,1)",
        "scale-in": "scale-in .22s cubic-bezier(.16,1,.3,1)",
        "pulse-dot": "pulse-dot 1.4s ease-in-out infinite",
        "logo-pulse": "logo-pulse 1.8s ease-in-out infinite",
        "sweep": "sweep 2.4s cubic-bezier(.4,0,.6,1) infinite",
        "count-up": "count-up .4s cubic-bezier(.16,1,.3,1) both",
      },
    },
  },
  plugins: [],
};
