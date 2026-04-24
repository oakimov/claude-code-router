import { useTranslation } from "react-i18next";
import React, { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { createPortal } from "react-dom";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Combobox } from "@/components/ui/combobox";
import { ColorPicker } from "@/components/ui/color-picker";
import { Badge } from "@/components/ui/badge";
import { useConfig } from "./ConfigProvider";
import {
  validateStatusLineConfig,
  formatValidationError,
  createDefaultStatusLineConfig,
} from "@/utils/statusline";
import type {
  StatusLineConfig,
  StatusLineModuleConfig,
  StatusLineThemeConfig,
} from "@/types";

const DEFAULT_MODULE: StatusLineModuleConfig = {
  type: "workDir",
  icon: "󰉋",
  text: "{{workDirName}}",
  color: "bright_blue",
};

// Nerd Font options
const NERD_FONTS = [
  { label: "Hack Nerd Font Mono", value: "Hack Nerd Font Mono" },
  { label: "FiraCode Nerd Font Mono", value: "FiraCode Nerd Font Mono" },
  {
    label: "JetBrainsMono Nerd Font Mono",
    value: "JetBrainsMono Nerd Font Mono",
  },
  { label: "Monaspace Nerd Font Mono", value: "Monaspace Nerd Font Mono" },
  { label: "UbuntuMono Nerd Font", value: "UbuntuMono Nerd Font" },
];

// Module type options
const MODULE_TYPES = [
  { label: "workDir", value: "workDir" },
  { label: "gitBranch", value: "gitBranch" },
  { label: "model", value: "model" },
  { label: "usage", value: "usage" },
  { label: "speed", value: "speed" },
  { label: "script", value: "script" },
];

// ANSI color code mapping
const ANSI_COLORS: Record<string, string> = {
  // Standard colors
  black: "text-black",
  red: "text-red-600",
  green: "text-green-600",
  yellow: "text-yellow-500",
  blue: "text-blue-500",
  magenta: "text-purple-500",
  cyan: "text-cyan-500",
  white: "text-white",
  // Bright colors
  bright_black: "text-gray-500",
  bright_red: "text-red-400",
  bright_green: "text-green-400",
  bright_yellow: "text-yellow-300",
  bright_blue: "text-blue-300",
  bright_magenta: "text-purple-300",
  bright_cyan: "text-cyan-300",
  bright_white: "text-white",
  // Background colors
  bg_black: "bg-black",
  bg_red: "bg-red-600",
  bg_green: "bg-green-600",
  bg_yellow: "bg-yellow-500",
  bg_blue: "bg-blue-500",
  bg_magenta: "bg-purple-500",
  bg_cyan: "bg-cyan-500",
  bg_white: "bg-white",
  // Bright background colors
  bg_bright_black: "bg-gray-800",
  bg_bright_red: "bg-red-400",
  bg_bright_green: "bg-green-400",
  bg_bright_yellow: "bg-yellow-300",
  bg_bright_blue: "bg-blue-300",
  bg_bright_magenta: "bg-purple-300",
  bg_bright_cyan: "bg-cyan-300",
  bg_bright_white: "bg-gray-100",
  // Additional background colors required for Powerline style
  bg_bright_orange: "bg-orange-400",
  bg_bright_purple: "bg-purple-400",
};


// Icon search input component
interface IconData {
  className: string;
  unicode: string;
  char: string;
}

interface IconSearchInputProps {
  value: string;
  onChange: (value: string) => void;
  fontFamily: string;
  t: (key: string) => string;
}

const IconSearchInput = React.memo(({ value, onChange, fontFamily, t }: IconSearchInputProps) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState(value);
  const [icons, setIcons] = useState<IconData[]>([]);
  const [filteredIcons, setFilteredIcons] = useState<IconData[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const inputRef = React.useRef<HTMLInputElement>(null);

  // Load Nerdfonts icon data
  const loadIcons = useCallback(async () => {
    if (icons.length > 0) return; // Already loaded
    
    setIsLoading(true);
    try {
      const response = await fetch('https://www.nerdfonts.com/assets/css/combo.css');
      const cssText = await response.text();
      
      // Parse icon class names and Unicode from CSS
      const iconRegex = /\.nf-([a-zA-Z0-9_-]+):before\s*\{\s*content:\s*"\\([0-9a-fA-F]+)";?\s*\}/g;
      const iconData: IconData[] = [];
      let match;
      
      while ((match = iconRegex.exec(cssText)) !== null) {
        const className = `nf-${match[1]}`;
        const unicode = match[2];
        const char = String.fromCharCode(parseInt(unicode, 16));
        iconData.push({ className, unicode, char });
      }
      
      setIcons(iconData);
      setFilteredIcons(iconData.slice(0, 200));
    } catch (error) {
      console.error('Failed to load icons:', error);
      setIcons([]);
      setFilteredIcons([]);
    } finally {
      setIsLoading(false);
    }
  }, [icons.length]);

  // Fuzzy search icons
  useEffect(() => {
    if (searchTerm.trim() === '') {
      setFilteredIcons(icons.slice(0, 100)); // Show first 100 icons
      return;
    }
    
    const term = searchTerm.toLowerCase();
    let filtered = icons;
    
    // If special characters are entered (possibly pasted icon), search for corresponding icon
    if (term.length === 1 || /[\u{2000}-\u{2FFFF}]/u.test(searchTerm)) {
      const pastedIcon = icons.find(icon => icon.char === searchTerm);
      if (pastedIcon) {
        filtered = [pastedIcon];
      } else {
        // Search for icons containing this character
        filtered = icons.filter(icon => icon.char === searchTerm);
      }
    } else {
      // Fuzzy search: class name, simplified name matching
      filtered = icons.filter(icon => {
        const className = icon.className.toLowerCase();
        const simpleClassName = className.replace(/[_-]/g, '');
        const simpleTerm = term.replace(/[_-]/g, '');
        
        return (
          className.includes(term) ||
          simpleClassName.includes(simpleTerm) ||
          // Keyword match
          term.split(' ').every(keyword => 
            className.includes(keyword) || simpleClassName.includes(keyword)
          )
        );
      });
    }
    
    setFilteredIcons(filtered.slice(0, 120)); // Show more results
  }, [searchTerm, icons]);

  // Handle input changes
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = e.target.value;
    setSearchTerm(newValue);
    onChange(newValue);
    
    // Always open dropdown to allow user to search or confirm pasted content
    setIsOpen(true);
    if (icons.length === 0) {
      loadIcons();
    }
  };

  // Handle paste event
  const handlePaste = (e: React.ClipboardEvent<HTMLInputElement>) => {
    const pastedText = e.clipboardData.getData('text');
    
    // If single character (possibly icon), accept directly and open dropdown to show corresponding icon
    if (pastedText && pastedText.length === 1) {
      setTimeout(() => {
        setIsOpen(true);
      }, 10);
    }
  };

  // Select icon
  const handleIconSelect = (iconChar: string) => {
    setSearchTerm(iconChar);
    onChange(iconChar);
    setIsOpen(false);
    inputRef.current?.focus();
  };

  // Handle focus event
  const handleFocus = () => {
    setIsOpen(true);
    if (icons.length === 0) {
      loadIcons();
    }
  };


  // Handle blur event (delayed close to allow clicking icon)
  const handleBlur = () => {
    setTimeout(() => setIsOpen(false), 200);
  };

  return (
    <div className="relative">
      <div className="relative">
        <Input
          ref={inputRef}
          value={searchTerm}
          onChange={handleInputChange}
          onFocus={handleFocus}
          onBlur={handleBlur}
          onPaste={handlePaste}
          placeholder={t("statusline.icon_placeholder")}
          style={{ fontFamily: fontFamily + ', monospace' }}
          className="text-lg pr-2"
        />
      </div>
      {isOpen && (
        <div className="absolute z-50 mt-1 w-full max-h-72 overflow-hidden rounded-md border bg-popover text-popover-foreground shadow-lg">
            {isLoading ? (
              <div className="flex items-center justify-center p-8">
                <svg className="animate-spin h-6 w-6 text-primary" viewBox="0 0 24 24">
                  <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" opacity="0.1"/>
                  <path fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                </svg>
              </div>
            ) : (
              <>
                <div className="grid grid-cols-5 gap-2 p-2 max-h-72 overflow-y-auto">
                  {filteredIcons.map((icon) => (
                    <div
                      key={icon.className}
                      className="flex items-center justify-center p-3 text-2xl cursor-pointer hover:bg-secondary rounded transition-colors"
                      onClick={() => handleIconSelect(icon.char)}
                      onMouseDown={(e) => e.preventDefault()} // Prevent loss of focus
                      title={`${icon.char} - ${icon.className}`}
                      style={{ fontFamily: fontFamily + ', monospace' }}
                    >
                      {icon.char}
                    </div>
                  ))}
                  {filteredIcons.length === 0 && (
                    <div className="col-span-5 flex flex-col items-center justify-center p-8 text-muted-foreground">
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="mb-2">
                        <circle cx="11" cy="11" r="8" />
                        <path d="m21 21-4.35-4.35" />
                      </svg>
                      <div className="text-sm">
                        {searchTerm ? `${t("statusline.no_icons_found")} "${searchTerm}"` : t("statusline.no_icons_available")}
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
      )}
    </div>
  );
});

// Variable replacement function
function replaceVariables(
  text: string,
  variables: Record<string, string>
): string {
  return text.replace(/\{\{(\w+)\}\}/g, (match, varName) => {
    return variables[varName] || match;
  });
}

// Render single module preview
function renderModulePreview(
  module: StatusLineModuleConfig,
  isPowerline: boolean = false
): React.ReactNode {
  // Mock variable data
  const variables = {
    workDirName: "project",
    gitBranch: "main",
    model: "Claude Sonnet 4",
    inputTokens: "1.2k",
    outputTokens: "2.5k",
  };

  const text = replaceVariables(module.text, variables);
  const icon = module.icon || "";

  // If text is empty and not usage type, skip this module
  if (!text && module.type !== "usage") {
    return null;
  }

  // Check if it's a hex color value
  const isHexColor = (color: string) => /^#[0-9A-F]{6}$/i.test(color);

  // If Powerline style, add background color and separator
  if (isPowerline) {
    // Handle background color - supports ANSI colors and hex colors
    let bgColorStyle = {};
    let bgColorClass = "";
    let separatorDataBg = "";
    if (module.background) {
      if (isHexColor(module.background)) {
        bgColorStyle = { backgroundColor: module.background };
        // For hex colors, we use the color value directly as data attribute
        separatorDataBg = module.background;
      } else {
        bgColorClass = ANSI_COLORS[module.background] || "";
        separatorDataBg = module.background;
      }
    }

    // Handle text color - supports ANSI colors and hex colors
    let textColorStyle = {};
    let textColorClass = "";
    if (module.color) {
      if (isHexColor(module.color)) {
        textColorStyle = { color: module.color };
      } else {
        textColorClass = ANSI_COLORS[module.color] || "text-white";
      }
    } else {
      textColorClass = "text-white";
    }

    return (
      <div
        className={`powerline-module px-4 ${bgColorClass} ${textColorClass}`}
        style={{ ...bgColorStyle, ...textColorStyle }}
      >
        <div className="powerline-module-content">
          {icon && <span>{icon}</span>}
          <span>{text}</span>
        </div>
        <div
          className="powerline-separator"
          data-current-bg={separatorDataBg}
        />
      </div>
    );
  }

  // Handle color under default style
  let textStyle = {};
  let textClass = "";
  if (module.color) {
    if (isHexColor(module.color)) {
      textStyle = { color: module.color };
    } else {
      textClass = ANSI_COLORS[module.color] || "";
    }
  }

  return (
    <>
      {icon && (
        <span style={textStyle} className={textClass}>
          {icon}
        </span>
      )}
      <span style={textStyle} className={textClass}>
        {text}
      </span>
    </>
  );
}

interface StatusLineConfigDialogProps {
  isOpen: boolean;
  onOpenChange: (isOpen: boolean) => void;
}

export function StatusLineConfigDialog({
  isOpen,
  onOpenChange,
}: StatusLineConfigDialogProps) {
  const { t } = useTranslation();
  const { config, setConfig } = useConfig();

  const [statusLineConfig, setStatusLineConfig] = useState<StatusLineConfig>(
    config?.StatusLine || createDefaultStatusLineConfig()
  );

  // Font status
  const [fontFamily, setFontFamily] = useState<string>(
    config?.StatusLine?.fontFamily || "Hack Nerd Font Mono"
  );

  const [selectedModuleIndex, setSelectedModuleIndex] = useState<number | null>(
    null
  );
  const [hexBackgroundColors, setHexBackgroundColors] = useState<Set<string>>(
    new Set()
  );

  // Add Powerline separator style
  useEffect(() => {
    const styleElement = document.createElement("style");
    styleElement.innerHTML = `
      .powerline-module {
        display: inline-flex;
        align-items: center;
        height: 28px;
        position: relative;
        padding: 0 8px;
        overflow: visible;
      }
      
      .powerline-module-content {
        display: flex;
        align-items: center;
        gap: 4px;
        position: relative;
      }
      
      .powerline-separator {
        width: 0;
        height: 0;
        border-top: 14px solid transparent;
        border-bottom: 14px solid transparent;
        border-left: 8px solid;
        position: absolute;
        right: -8px;
        top: 0;
        display: block;
      }
      
      /* Use z-index to ensure each module's triangle overlaps the next module on top */
      .cursor-pointer:nth-child(1) .powerline-separator { z-index: 10; }
      .cursor-pointer:nth-child(2) .powerline-separator { z-index: 9; }
      .cursor-pointer:nth-child(3) .powerline-separator { z-index: 8; }
      .cursor-pointer:nth-child(4) .powerline-separator { z-index: 7; }
      .cursor-pointer:nth-child(5) .powerline-separator { z-index: 6; }
      .cursor-pointer:nth-child(6) .powerline-separator { z-index: 5; }
      .cursor-pointer:nth-child(7) .powerline-separator { z-index: 4; }
      .cursor-pointer:nth-child(8) .powerline-separator { z-index: 3; }
      .cursor-pointer:nth-child(9) .powerline-separator { z-index: 2; }
      .cursor-pointer:nth-child(10) .powerline-separator { z-index: 1; }
      
      .cursor-pointer:last-child .powerline-separator {
        display: none;
      }
      
      /* Dynamically set color based on data attribute to ensure consistency with module background color */
      .powerline-separator[data-current-bg="bg_black"] { border-left-color: #000000; }
      .powerline-separator[data-current-bg="bg_red"] { border-left-color: #dc2626; }
      .powerline-separator[data-current-bg="bg_green"] { border-left-color: #16a34a; }
      .powerline-separator[data-current-bg="bg_yellow"] { border-left-color: #eab308; }
      .powerline-separator[data-current-bg="bg_blue"] { border-left-color: #3b82f6; }
      .powerline-separator[data-current-bg="bg_magenta"] { border-left-color: #a855f7; }
      .powerline-separator[data-current-bg="bg_cyan"] { border-left-color: #06b6d4; }
      .powerline-separator[data-current-bg="bg_white"] { border-left-color: #ffffff; }
      .powerline-separator[data-current-bg="bg_bright_black"] { border-left-color: #1f2937; }
      .powerline-separator[data-current-bg="bg_bright_red"] { border-left-color: #f87171; }
      .powerline-separator[data-current-bg="bg_bright_green"] { border-left-color: #4ade80; }
      .powerline-separator[data-current-bg="bg_bright_yellow"] { border-left-color: #fde047; }
      .powerline-separator[data-current-bg="bg_bright_blue"] { border-left-color: #93c5fd; }
      .powerline-separator[data-current-bg="bg_bright_magenta"] { border-left-color: #c084fc; }
      .powerline-separator[data-current-bg="bg_bright_cyan"] { border-left-color: #22d3ee; }
      .powerline-separator[data-current-bg="bg_bright_white"] { border-left-color: #f3f4f6; }
      .powerline-separator[data-current-bg="bg_bright_orange"] { border-left-color: #fb923c; }
      .powerline-separator[data-current-bg="bg_bright_purple"] { border-left-color: #c084fc; }
    `;
    document.head.appendChild(styleElement);

    // Cleanup function
    return () => {
      document.head.removeChild(styleElement);
    };
  }, []);

  // Dynamically update hex background color styles
  useEffect(() => {
    // Collect hex background colors used in all modules
    const hexColors = new Set<string>();
    Object.keys(statusLineConfig).forEach((key) => {
      const themeConfig = statusLineConfig[key as keyof StatusLineConfig];
      if (
        themeConfig &&
        typeof themeConfig === "object" &&
        "modules" in themeConfig
      ) {
        const modules = (themeConfig as StatusLineThemeConfig).modules || [];
        modules.forEach((module) => {
          if (module.background && /^#[0-9A-F]{6}$/i.test(module.background)) {
            hexColors.add(module.background);
          }
        });
      }
    });

    setHexBackgroundColors(hexColors);

    // Create dynamic style element
    const styleElement = document.createElement("style");
    styleElement.id = "hex-powerline-styles";

    // Generate CSS rules for hex colors
    let cssRules = "";
    hexColors.forEach((color) => {
      // Convert hex color to RGB value
      const r = parseInt(color.slice(1, 3), 16);
      const g = parseInt(color.slice(3, 5), 16);
      const b = parseInt(color.slice(5, 7), 16);
      cssRules += `.powerline-separator[data-current-bg="${color}"] { border-left-color: rgb(${r}, ${g}, ${b}); }\n`;
    });

    styleElement.innerHTML = cssRules;
    document.head.appendChild(styleElement);

    // Cleanup function
    return () => {
      const existingStyle = document.getElementById("hex-powerline-styles");
      if (existingStyle) {
        document.head.removeChild(existingStyle);
      }
    };
  }, [statusLineConfig]);

  // Module type options
  const MODULE_TYPES_OPTIONS = MODULE_TYPES.map((item) => ({
    ...item,
    label: t(`statusline.${item.label}`),
  }));

  const handleThemeChange = (value: string) => {
    setStatusLineConfig((prev) => ({ ...prev, currentStyle: value }));
  };

  const handleModuleChange = (
    index: number,
    field: keyof StatusLineModuleConfig,
    value: string
  ) => {
    const currentTheme =
      statusLineConfig.currentStyle as keyof StatusLineConfig;
    const themeConfig = statusLineConfig[currentTheme];
    const modules =
      themeConfig && typeof themeConfig === "object" && "modules" in themeConfig
        ? [...((themeConfig as StatusLineThemeConfig).modules || [])]
        : [];
    if (modules[index]) {
      modules[index] = { ...modules[index], [field]: value };
    }

    setStatusLineConfig((prev) => ({
      ...prev,
      [currentTheme]: { modules },
    }));
  };

  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const handleSave = () => {
    // Validate configuration
    const validationResult = validateStatusLineConfig(statusLineConfig);

    if (!validationResult.isValid) {
      // Format error messages
      const errorMessages = validationResult.errors.map((error) =>
        formatValidationError(error, t)
      );
      setValidationErrors(errorMessages);
      return;
    }

    // Clear previous errors
    setValidationErrors([]);

    if (config) {
      setConfig({
        ...config,
        StatusLine: {
          ...statusLineConfig,
          fontFamily,
        },
      });
      onOpenChange(false);
    }
  };

  // Create custom Alert component
  const CustomAlert = ({
    title,
    description,
    variant = "default",
  }: {
    title: string;
    description: React.ReactNode;
    variant?: "default" | "destructive";
  }) => {
    const isError = variant === "destructive";

    return (
      <div
        className={`rounded-lg border p-4 ${
          isError
            ? "bg-red-50 border-red-200 text-red-800"
            : "bg-blue-50 border-blue-200 text-blue-800"
        }`}
      >
        <div className="flex">
          <div className="flex-shrink-0">
            {isError ? (
              <svg
                className="h-5 w-5 text-red-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
            ) : (
              <svg
                className="h-5 w-5 text-blue-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                  clipRule="evenodd"
                />
              </svg>
            )}
          </div>
          <div className="ml-3">
            <h3
              className={`text-sm font-medium ${
                isError ? "text-red-800" : "text-blue-800"
              }`}
            >
              {title}
            </h3>
            <div
              className={`mt-2 text-sm ${
                isError ? "text-red-700" : "text-blue-700"
              }`}
            >
              {description}
            </div>
          </div>
        </div>
      </div>
    );
  };

  const currentThemeKey =
    statusLineConfig.currentStyle as keyof StatusLineConfig;
  const currentThemeConfig = statusLineConfig[currentThemeKey];
  const currentModules =
    currentThemeConfig &&
    typeof currentThemeConfig === "object" &&
    "modules" in currentThemeConfig
      ? (currentThemeConfig as StatusLineThemeConfig).modules || []
      : [];
  const selectedModule =
    selectedModuleIndex !== null && currentModules.length > selectedModuleIndex
      ? currentModules[selectedModuleIndex]
      : null;

  // Function to delete selected module
  const deleteSelectedModule = useCallback(() => {
    if (selectedModuleIndex === null) return;
    
    const currentTheme =
      statusLineConfig.currentStyle as keyof StatusLineConfig;
    const themeConfig = statusLineConfig[currentTheme];
    const modules =
      themeConfig &&
      typeof themeConfig === "object" &&
      "modules" in themeConfig
        ? [...((themeConfig as StatusLineThemeConfig).modules || [])]
        : [];
        
    if (selectedModuleIndex >= 0 && selectedModuleIndex < modules.length) {
      modules.splice(selectedModuleIndex, 1);
      
      setStatusLineConfig((prev) => ({
        ...prev,
        [currentTheme]: { modules },
      }));
      
      setSelectedModuleIndex(null);
    }
  }, [selectedModuleIndex, statusLineConfig]);

  // Font styles
  const fontStyle = fontFamily ? { fontFamily } : {};

  // Keyboard event listener, supports deleting selected module
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Check if module is selected
      if (selectedModuleIndex === null) return;

      // Check if Delete or Backspace key is pressed
      if (e.key === 'Delete' || e.key === 'Backspace') {
        // Check if focus element is a module in the preview area
        const activeElement = document.activeElement as HTMLElement;

        // Check if focus element is a module in the preview area (has cursor-pointer class and tabIndex)
        const isPreviewModule = activeElement?.classList.contains('cursor-pointer') &&
                               activeElement?.hasAttribute('tabIndex');

        // Only perform delete operation when focus is on a component in the preview area
        if (isPreviewModule) {
          e.preventDefault();
          deleteSelectedModule();
        }
      }
    };

    // Add event listener
    document.addEventListener('keydown', handleKeyDown);

    // Cleanup function
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [selectedModuleIndex, deleteSelectedModule]);

  // Force re-render when font or theme changes
  const fontKey = `${fontFamily}-${statusLineConfig.currentStyle}`;

  return (
    <Dialog open={isOpen} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl h-[90vh] overflow-hidden sm:max-w-5xl md:max-w-6xl lg:max-w-7xl animate-in fade-in-90 slide-in-from-bottom-10 duration-300 flex flex-col">
        <DialogHeader
          data-testid="statusline-config-dialog-header"
          className="border-b pb-4"
        >
          <DialogTitle className="flex items-center">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="mr-2"
            >
              <path d="M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
              <path d="M14 3v4a2 2 0 0 0 2 2h4" />
              <path d="M3 12h18" />
            </svg>
            {t("statusline.title")}
          </DialogTitle>
        </DialogHeader>

        {/* Error display area */}
        {validationErrors.length > 0 && (
          <div className="px-6">
            <CustomAlert
              variant="destructive"
              title="Configuration validation failed"
              description={
                <ul className="list-disc pl-5 space-y-1">
                  {validationErrors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              }
            />
          </div>
        )}

        <div className="flex flex-col gap-6 flex-1 overflow-hidden">
          {/* Configuration panel */}
          <div className="space-y-6">
            {/* Theme style and font selection */}
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="theme-style" className="text-sm font-medium">
                  {t("statusline.theme")}
                </Label>
                <Combobox
                  options={[
                    { label: t("statusline.theme_default"), value: "default" },
                    { label: t("statusline.theme_powerline"), value: "powerline" },
                  ]}
                  value={statusLineConfig.currentStyle}
                  onChange={handleThemeChange}
                  data-testid="theme-selector"
                  placeholder={t("statusline.theme_placeholder")}
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="font-family" className="text-sm font-medium">
                  {t("statusline.module_icon")}
                </Label>
                <Combobox
                  options={NERD_FONTS}
                  value={fontFamily}
                  onChange={(value) => setFontFamily(value)}
                  data-testid="font-family-selector"
                  placeholder={t("statusline.font_placeholder")}
                />
              </div>
            </div>
          </div>

          {/* Three-column layout: Component list | Preview area | Property configuration */}
          <div className="grid grid-cols-5 gap-6 overflow-hidden flex-1">
            {/* Left: Supported components */}
            <div className="border rounded-lg flex flex-col overflow-hidden col-span-1">
              <h3 className="text-sm font-medium p-4 pb-0 mb-3">{t("statusline.components")}</h3>
              <div className="space-y-2 overflow-y-auto px-4 pb-4 flex-1">
                {MODULE_TYPES_OPTIONS.map((moduleType) => (
                  <div
                    key={moduleType.value}
                    className="flex items-center gap-2 p-2 border rounded cursor-move hover:bg-secondary"
                    draggable
                    onDragStart={(e) => {
                      e.dataTransfer.setData("moduleType", moduleType.value);
                    }}
                  >
                    <span className="text-sm">{moduleType.label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Middle: Preview area */}
            <div className="border rounded-lg p-4 flex flex-col col-span-3">
              <h3 className="text-sm font-medium mb-3">{t("statusline.preview")}</h3>
              <div
                key={fontKey}
                className={`rounded bg-black/90 text-white font-mono text-sm overflow-x-auto flex items-center border border-border p-3 py-5 shadow-inner overflow-hidden ${
                  statusLineConfig.currentStyle === "powerline"
                    ? "gap-0 h-8 p-0 items-center relative"
                    : "h-5"
                }`}
                data-testid="statusline-preview"
                style={fontStyle}
                onDragOver={(e) => {
                  e.preventDefault();
                }}
                onDrop={(e) => {
                  e.preventDefault();
                  const moduleType = e.dataTransfer.getData("moduleType");
                  if (moduleType) {
                    // Add new module
                    const currentTheme =
                      statusLineConfig.currentStyle as keyof StatusLineConfig;
                    const themeConfig = statusLineConfig[currentTheme];
                    const modules =
                      themeConfig &&
                      typeof themeConfig === "object" &&
                      "modules" in themeConfig
                        ? [
                            ...((themeConfig as StatusLineThemeConfig)
                              .modules || []),
                          ]
                        : [];

                    // Set default values based on module type
                    let newModule: StatusLineModuleConfig;
                    switch (moduleType) {
                      case "workDir":
                        newModule = {
                          type: "workDir",
                          icon: "󰉋",
                          text: "{{workDirName}}",
                          color: "bright_blue",
                        };
                        break;
                      case "gitBranch":
                        newModule = {
                          type: "gitBranch",
                          icon: "🌿",
                          text: "{{gitBranch}}",
                          color: "bright_green",
                        };
                        break;
                      case "model":
                        newModule = {
                          type: "model",
                          icon: "🤖",
                          text: "{{model}}",
                          color: "bright_yellow",
                        };
                        break;
                      case "usage":
                        newModule = {
                          type: "usage",
                          icon: "📊",
                          text: "{{inputTokens}} → {{outputTokens}}",
                          color: "bright_magenta",
                        };
                        break;
                      case "speed":
                        newModule = {
                          type: "speed",
                          icon: "⚡",
                          text: "{{tokenSpeed}}",
                          color: "bright_green",
                        };
                        break;
                      case "script":
                        newModule = {
                          type: "script",
                          icon: "📜",
                          text: "Script Module",
                          color: "bright_cyan",
                          scriptPath: "",
                        };
                        break;
                      default:
                        newModule = { ...DEFAULT_MODULE, type: moduleType };
                    }

                    modules.push(newModule);

                    setStatusLineConfig((prev) => ({
                      ...prev,
                      [currentTheme]: { modules },
                    }));
                  }
                }}
              >
                {currentModules.length > 0 ? (
                  <div className="flex items-center flex-wrap gap-0">
                    {currentModules.map((module, index) => (
                      <div
                        key={index}
                        tabIndex={0}
                        className={`cursor-pointer ${
                          selectedModuleIndex === index
                            ? "bg-white/20"
                            : "hover:bg-white/10"
                        } ${
                          statusLineConfig.currentStyle === "powerline"
                            ? "p-0 rounded-none inline-flex overflow-visible relative"
                            : "flex items-center gap-1 px-2 py-1 rounded"
                        }`}
                        onClick={() => setSelectedModuleIndex(index)}
                        draggable
                        onDragStart={(e) => {
                          e.dataTransfer.setData("dragIndex", index.toString());
                        }}
                        onDragOver={(e) => {
                          e.preventDefault();
                        }}
                        onDrop={(e) => {
                          e.preventDefault();
                          const dragIndex = parseInt(
                            e.dataTransfer.getData("dragIndex")
                          );
                          if (!isNaN(dragIndex) && dragIndex !== index) {
                            // Reorder modules
                            const currentTheme =
                              statusLineConfig.currentStyle as keyof StatusLineConfig;
                            const themeConfig = statusLineConfig[currentTheme];
                            const modules =
                              themeConfig &&
                              typeof themeConfig === "object" &&
                              "modules" in themeConfig
                                ? [
                                    ...((themeConfig as StatusLineThemeConfig)
                                      .modules || []),
                                  ]
                                : [];

                            if (
                              dragIndex >= 0 &&
                              dragIndex < modules.length &&
                              index >= 0 &&
                              index <= modules.length
                            ) {
                              const [movedModule] = modules.splice(
                                dragIndex,
                                1
                              );
                              modules.splice(index, 0, movedModule);

                              setStatusLineConfig((prev) => ({
                                ...prev,
                                [currentTheme]: { modules },
                              }));

                              // Update selected item index
                              if (selectedModuleIndex === dragIndex) {
                                setSelectedModuleIndex(index);
                              } else if (selectedModuleIndex === index) {
                                setSelectedModuleIndex(dragIndex);
                              }
                            }
                          }
                        }}
                      >
                        {renderModulePreview(
                          module,
                          statusLineConfig.currentStyle === "powerline"
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center w-full py-4 text-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="text-gray-500 mb-2"
                    >
                      <path d="M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18z" />
                      <path d="M12 8v8" />
                      <path d="M8 12h8" />
                    </svg>
                    <span className="text-gray-500 text-sm">
                      {t("statusline.drag_hint")}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Right: Property configuration */}
            <div className="border rounded-lg flex flex-col overflow-hidden col-span-1">
              <h3 className="text-sm font-medium p-4 pb-0 mb-3">{t("statusline.properties")}</h3>
              <div className="overflow-y-auto px-4 pb-4 flex-1">
                {selectedModule && selectedModuleIndex !== null ? (
                  <div className="space-y-4">
                    
                    <div className="space-y-2">
                      <Label htmlFor="module-icon">
                        {t("statusline.module_icon")}
                      </Label>
                      <IconSearchInput
                        key={fontKey}
                        value={selectedModule.icon || ""}
                        onChange={(value) =>
                          handleModuleChange(
                            selectedModuleIndex,
                            "icon",
                            value
                          )
                        }
                        fontFamily={fontFamily}
                        t={t}
                      />
                      <p className="text-xs text-muted-foreground">
                        {t("statusline.icon_description")}
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="module-text">
                        {t("statusline.module_text")}
                      </Label>
                      <Input
                        id="module-text"
                        value={selectedModule.text}
                        onChange={(e) =>
                          handleModuleChange(
                            selectedModuleIndex,
                            "text",
                            e.target.value
                          )
                        }
                        placeholder={t("statusline.text_placeholder")}
                      />
                      <div className="text-xs text-muted-foreground">
                        <p>{t("statusline.module_text_description")}</p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          <Badge
                            variant="secondary"
                            className="text-xs py-0.5 px-1.5"
                          >
                            {"{{workDirName}}"}
                          </Badge>
                          <Badge
                            variant="secondary"
                            className="text-xs py-0.5 px-1.5"
                          >
                            {"{{gitBranch}}"}
                          </Badge>
                          <Badge
                            variant="secondary"
                            className="text-xs py-0.5 px-1.5"
                          >
                            {"{{model}}"}
                          </Badge>
                          <Badge
                            variant="secondary"
                            className="text-xs py-0.5 px-1.5"
                          >
                            {"{{inputTokens}}"}
                          </Badge>
                          <Badge
                            variant="secondary"
                            className="text-xs py-0.5 px-1.5"
                          >
                            {"{{outputTokens}}"}
                          </Badge>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label>{t("statusline.module_color")}</Label>
                      <ColorPicker
                        value={selectedModule.color || ""}
                        onChange={(value) =>
                          handleModuleChange(
                            selectedModuleIndex,
                            "color",
                            value
                          )
                        }
                      />
                      <p className="text-xs text-muted-foreground">
                        {t("statusline.module_color_description")}
                      </p>
                    </div>

                    <div className="space-y-2">
                      <Label>{t("statusline.module_background")}</Label>
                      <ColorPicker
                        value={selectedModule.background || ""}
                        onChange={(value) =>
                          handleModuleChange(
                            selectedModuleIndex,
                            "background",
                            value
                          )
                        }
                      />
                      <p className="text-xs text-muted-foreground">
                        {t("statusline.module_background_description")}
                      </p>
                    </div>

                    {/* Script Path input field - only show when type is script */}
                    {selectedModule.type === "script" && (
                      <div className="space-y-2">
                        <Label htmlFor="module-script-path">
                          {t("statusline.module_script_path")}
                        </Label>
                        <Input
                          id="module-script-path"
                          value={selectedModule.scriptPath || ""}
                          onChange={(e) =>
                            handleModuleChange(
                              selectedModuleIndex,
                              "scriptPath",
                              e.target.value
                            )
                          }
                          placeholder={t("statusline.script_placeholder")}
                        />
                        <p className="text-xs text-muted-foreground">
                          {t("statusline.module_script_path_description")}
                        </p>
                      </div>
                    )}

                    
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={deleteSelectedModule}
                    >
                      {t("statusline.delete_module")}
                    </Button>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full min-h-[200px]">
                    <p className="text-muted-foreground text-sm">
                      {t("statusline.select_hint")}
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <DialogFooter className="border-t pt-4 mt-4">
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            className="transition-all hover:scale-105"
          >
            {t("app.cancel")}
          </Button>
          <Button
            onClick={handleSave}
            data-testid="save-statusline-config"
            className="transition-all hover:scale-105"
          >
            {t("app.save")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
