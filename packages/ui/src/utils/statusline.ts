import type { StatusLineConfig, StatusLineModuleConfig } from "@/types";

// Validation results (keep interface but not used)
export interface ValidationResult {
  isValid: boolean;
  errors: any[];
}

/**
 * Validate StatusLine configuration - all validation removed
 * @param config the configuration object to validate
 * @returns always returns validation success
 */
export function validateStatusLineConfig(config: unknown): ValidationResult {
  // No longer performing any validation
  return { isValid: true, errors: [] };
}


/**
 * Format error message (supports i18n) - no longer used
 */
export function formatValidationError(error: unknown, t: (key: string, options?: Record<string, unknown>) => string): string {
  return t("statusline.validation.unknown_error");
}

/**
 * Parse color value, supports hexadecimal and built-in color names
 * @param color color value (can be color name or hex value)
 * @param defaultColor default color (hexadecimal)
 * @returns hexadecimal color value
 */
export function parseColorValue(color: string | undefined, defaultColor: string = "#ffffff"): string {
  if (!color) {
    return defaultColor;
  }
  
  // If it's a hex color value (starts with #)
  if (color.startsWith('#')) {
    return color;
  }
  
  // If it's a known color name, return corresponding hex value
  return COLOR_HEX_MAP[color] || defaultColor;
}

/**
 * Check if it's a valid hexadecimal color value
 * @param color color value to check
 * @returns whether it's a valid hexadecimal color value
 */
export function isHexColor(color: string): boolean {
  return /^#([0-9A-F]{3}){1,2}$/i.test(color);
}

// Mapping from color enum to hex
export const COLOR_HEX_MAP: Record<string, string> = {
  black: "#000000",
  red: "#cd0000",
  green: "#00cd00",
  yellow: "#cdcd00",
  blue: "#0000ee",
  magenta: "#cd00cd",
  cyan: "#00cdcd",
  white: "#e5e5e5",
  bright_black: "#7f7f7f",
  bright_red: "#ff0000",
  bright_green: "#00ff00",
  bright_yellow: "#ffff00",
  bright_blue: "#5c5cff",
  bright_magenta: "#ff00ff",
  bright_cyan: "#00ffff",
  bright_white: "#ffffff",
  bg_black: "#000000",
  bg_red: "#cd0000",
  bg_green: "#00cd00",
  bg_yellow: "#cdcd00",
  bg_blue: "#0000ee",
  bg_magenta: "#cd00cd",
  bg_cyan: "#00cdcd",
  bg_white: "#e5e5e5",
  bg_bright_black: "#7f7f7f",
  bg_bright_red: "#ff0000",
  bg_bright_green: "#00ff00",
  bg_bright_yellow: "#ffff00",
  bg_bright_blue: "#5c5cff",
  bg_bright_magenta: "#ff00ff",
  bg_bright_cyan: "#00ffff",
  bg_bright_white: "#ffffff"
};

/**
 * Create default StatusLine configuration
 */
export function createDefaultStatusLineConfig(): StatusLineConfig {
  return {
    enabled: false,
    currentStyle: "default",
    default: { 
      modules: [
        { type: "workDir", icon: "󰉋", text: "{{workDirName}}", color: "bright_blue" },
        { type: "gitBranch", icon: "", text: "{{gitBranch}}", color: "bright_magenta" },
        { type: "model", icon: "󰚩", text: "{{model}}", color: "bright_cyan" },
        { type: "usage", icon: "↑", text: "{{inputTokens}}", color: "bright_green" },
        { type: "usage", icon: "↓", text: "{{outputTokens}}", color: "bright_yellow" }
      ] 
    },
    powerline: { 
      modules: [
        { type: "workDir", icon: "󰉋", text: "{{workDirName}}", color: "white", background: "bg_bright_blue" },
        { type: "gitBranch", icon: "", text: "{{gitBranch}}", color: "white", background: "bg_bright_magenta" },
        { type: "model", icon: "󰚩", text: "{{model}}", color: "white", background: "bg_bright_cyan" },
        { type: "usage", icon: "↑", text: "{{inputTokens}}", color: "white", background: "bg_bright_green" },
        { type: "usage", icon: "↓", text: "{{outputTokens}}", color: "white", background: "bg_bright_yellow" }
      ] 
    }
  };
}

/**
 * Create configuration backup
 */
export function backupConfig(config: StatusLineConfig): string {
  const backup = {
    config,
    timestamp: new Date().toISOString(),
    version: "1.0"
  };
  return JSON.stringify(backup, null, 2);
}

/**
 * Restore configuration from backup
 */
export function restoreConfig(backupStr: string): StatusLineConfig | null {
  try {
    const backup = JSON.parse(backupStr);
    if (backup && backup.config && backup.timestamp) {
      return backup.config as StatusLineConfig;
    }
    return null;
  } catch (error) {
    console.error("Failed to restore config from backup:", error);
    return null;
  }
}
