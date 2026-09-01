import { useEffect, useState } from "react"

type Theme = "dark" | "light"
const KEY = "vh-theme"

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(
    () => (localStorage.getItem(KEY) as Theme) || "dark",
  )
  useEffect(() => {
    const root = document.documentElement
    root.classList.toggle("dark", theme === "dark")
    localStorage.setItem(KEY, theme)
  }, [theme])
  return {
    theme,
    toggle: () => setTheme((t) => (t === "dark" ? "light" : "dark")),
  }
}
