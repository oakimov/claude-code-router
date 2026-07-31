import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig([
  globalIgnores(["**/dist/**", "**/node_modules/**"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    languageOptions: {
      globals: globals.node,
    },
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-require-imports": "off",
      "@typescript-eslint/no-unused-vars": "off",
      "@typescript-eslint/ban-ts-comment": "off",
      "@typescript-eslint/no-this-alias": "off",
      "no-control-regex": "off",
      "no-empty": ["error", { allowEmptyCatch: true }],
      "no-useless-assignment": "off",
      "no-useless-catch": "off",
      "no-useless-escape": "off",
      "prefer-const": "error",
      "preserve-caught-error": "off",
    },
  },
]);
