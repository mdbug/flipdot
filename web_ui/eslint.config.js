import js from "@eslint/js";
import globals from "globals";

export default [
  {
    ignores: ["vendor/**", "node_modules/**", "eslint.config.js", "tests/**"],
  },
  js.configs.recommended,
  {
    files: ["**/*.js"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "script",
      globals: {
        ...globals.browser,
        // Vendored / CDN libraries loaded via <script> tags.
        marked: "readonly",
        DOMPurify: "readonly",
        Chart: "readonly",
      },
    },
    rules: {
      "no-unused-vars": ["warn", { argsIgnorePattern: "^_", caughtErrorsIgnorePattern: "^_" }],
    },
  },
];
