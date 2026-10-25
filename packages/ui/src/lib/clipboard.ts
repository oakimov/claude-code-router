/**
 * `navigator.clipboard` is exposed only in secure contexts, so it is missing
 * when the UI is opened over plain HTTP on a non-localhost host. Fall back to a
 * hidden textarea plus the legacy copy command, which works in every context.
 */
export async function copyText(text: string): Promise<void> {
  if (typeof navigator !== "undefined" && typeof navigator.clipboard?.writeText === "function") {
    await navigator.clipboard.writeText(text);
    return;
  }
  if (legacyCopy(text)) return;
  throw new Error("Clipboard is unavailable");
}

function legacyCopy(text: string): boolean {
  if (typeof document === "undefined" || !document.body) return false;
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.top = "0";
  area.style.left = "0";
  area.style.opacity = "0";
  const selection = document.getSelection();
  const previous = selection && selection.rangeCount > 0 ? selection.getRangeAt(0) : null;
  document.body.appendChild(area);
  try {
    area.select();
    area.setSelectionRange(0, area.value.length);
    return typeof document.execCommand === "function" && document.execCommand("copy");
  } catch {
    return false;
  } finally {
    area.remove();
    if (selection && previous) {
      selection.removeAllRanges();
      selection.addRange(previous);
    }
  }
}
