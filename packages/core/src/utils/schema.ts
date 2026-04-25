/**
 * Valid JSON Schema fields. Unknown fields are stripped to prevent
 * API rejections from providers like Gemini and Mistral.
 */
const VALID_SCHEMA_FIELDS = new Set([
  "type",
  "format",
  "title",
  "description",
  "nullable",
  "enum",
  "maxItems",
  "minItems",
  "properties",
  "required",
  "minProperties",
  "maxProperties",
  "minLength",
  "maxLength",
  "pattern",
  "example",
  "anyOf",
  "propertyOrdering",
  "default",
  "items",
  "minimum",
  "maximum",
  "additionalProperties",
  "allOf",
  "oneOf",
]);

/**
 * Recursively removes format: 'uri' from JSON schemas.
 * Some providers (Gemini, Mistral) reject schemas with this format.
 */
function removeUriFormat(schema: any): any {
  if (!schema || typeof schema !== "object") return schema;

  if (schema.type === "string" && schema.format === "uri") {
    const { format, ...rest } = schema;
    return rest;
  }

  if (Array.isArray(schema)) {
    return schema.map((item: any) => removeUriFormat(item));
  }

  const result: any = {};
  for (const key in schema) {
    if (key === "properties") {
      result[key] = {};
      for (const propKey in schema[key]) {
        result[key][propKey] = removeUriFormat(schema[key][propKey]);
      }
    } else if (key === "items") {
      result[key] = removeUriFormat(schema[key]);
    } else if (
      ["anyOf", "allOf", "oneOf"].includes(key) &&
      Array.isArray(schema[key])
    ) {
      result[key] = schema[key].map((item: any) => removeUriFormat(item));
    } else {
      result[key] = removeUriFormat(schema[key]);
    }
  }
  return result;
}

/**
 * Recursively sanitizes a JSON schema by:
 * 1. Stripping unknown fields (not in VALID_SCHEMA_FIELDS)
 * 2. Removing enum from non-string types
 * 3. Removing format from string types (except "enum" and "date-time")
 * 4. Removing format: 'uri' from any type
 * 5. Removing $schema field
 *
 * This is a superset of normalizeJsonSchema that also handles
 * field whitelisting and enum/format cleanup needed by providers
 * like Gemini and Mistral.
 */
export function sanitizeJsonSchema(
  schema: any,
  parentKey?: string
): any {
  if (!schema || typeof schema !== "object") return schema;

  if (Array.isArray(schema)) {
    return schema.map((item: any) => sanitizeJsonSchema(item, parentKey));
  }

  const result: any = {};

  // Step 1: Strip unknown fields (preserve all keys inside "properties")
  for (const key in schema) {
    if (parentKey !== "properties" && !VALID_SCHEMA_FIELDS.has(key)) {
      continue;
    }

    let value = schema[key];

    // Step 2: Remove enum from non-string types
    if (key === "enum" && schema.type && schema.type !== "string") {
      continue;
    }

    // Step 3: Remove format from string types (except enum/date-time)
    if (
      key === "format" &&
      schema.type === "string" &&
      value &&
      !["enum", "date-time"].includes(value)
    ) {
      continue;
    }

    // Step 4: Remove format: 'uri'
    if (key === "format" && value === "uri") {
      continue;
    }

    // Step 5: Remove $schema
    if (key === "$schema") {
      continue;
    }

    // Recurse into nested structures
    if (key === "properties" && typeof value === "object" && !Array.isArray(value)) {
      const props: any = {};
      for (const propKey in value) {
        props[propKey] = sanitizeJsonSchema(value[propKey]);
      }
      result[key] = props;
    } else if (key === "items") {
      result[key] = sanitizeJsonSchema(value, key);
    } else if (
      ["anyOf", "allOf", "oneOf"].includes(key) &&
      Array.isArray(value)
    ) {
      result[key] = value.map((item: any) => sanitizeJsonSchema(item, key));
    } else {
      result[key] = value;
    }
  }

  return result;
}

/**
 * Normalizes a JSON schema by removing fields that may cause API rejections:
 * - format: 'uri' from string types
 * - $schema field
 *
 * For full sanitization (field whitelist, enum/format cleanup), use sanitizeJsonSchema instead.
 */
export function normalizeJsonSchema(schema: any): any {
  if (!schema || typeof schema !== "object") return schema;

  const normalized = removeUriFormat(schema);

  if (normalized.$schema) {
    delete normalized.$schema;
  }

  return normalized;
}

/**
 * Normalizes tool function parameters by removing problematic fields.
 * Uses sanitizeJsonSchema for thorough cleanup.
 */
export function normalizeToolParameters(parameters: any): any {
  if (!parameters || typeof parameters !== "object") return parameters;

  return sanitizeJsonSchema(parameters);
}
