import { useTheme } from "@/components/theme-provider";

/** Monaco editor theme that follows the app light/dark preference. */
export function useMonacoTheme(): "vs" | "vs-dark" {
  const { resolvedTheme } = useTheme();
  return resolvedTheme === "dark" ? "vs-dark" : "vs";
}
