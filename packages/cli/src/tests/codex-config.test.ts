/**
 * ccr codex-config: argument parsing, wildcard selection, catalog entry shape,
 * and idempotent managed-block editing of Codex's config.toml.
 *
 * Hermetic: no network (models.dev is not consulted by the pure helpers under
 * test) and no writes outside the assertions below.
 */
import assert from "node:assert/strict";
import {
  parseCodexConfigArgs,
  globToRegExp,
  matchesModel,
  selectModels,
  buildCatalogEntry,
  patchCodexConfig,
  type CatalogModel,
} from "../utils/codexConfig";

const CONFIG = {
  PORT: 3456,
  Providers: [
    { name: "deepseek", models: ["deepseek-chat", "deepseek-reasoner"] },
    { name: "openai", models: ["gpt-5.4", "gpt-5-mini", "o3"] },
    { name: "empty", models: [] },
  ],
};

function main(): void {
  // ── argument parsing ──────────────────────────────────────────────
  {
    const opts = parseCodexConfigArgs([
      "--providers",
      "deepseek, openai",
      "--models",
      "gpt-*,o3",
      "--dry-run",
    ]);
    assert.deepEqual(opts.providers, ["deepseek", "openai"]);
    assert.deepEqual(opts.models, ["gpt-*", "o3"]);
    assert.equal(opts.dryRun, true);
    assert.equal(opts.force, false);
    assert.equal(opts.codexProbe, true);
  }

  {
    const opts = parseCodexConfigArgs([]);
    assert.deepEqual(opts.models, ["*"]);
    assert.equal(opts.providers, undefined);
  }

  {
    assert.throws(() => parseCodexConfigArgs(["--models"]), /requires a value/);
    assert.throws(
      () => parseCodexConfigArgs(["--models", "--dry-run"]),
      /--models requires a value/
    );
    assert.throws(() => parseCodexConfigArgs(["--bogus"]), /Unknown option/);
  }

  // ── wildcard matching ─────────────────────────────────────────────
  {
    assert.ok(globToRegExp("gpt-*").test("gpt-5.4"));
    assert.ok(!globToRegExp("gpt-*").test("claude-3"));
    // Dots are literal, not "any character".
    assert.ok(!globToRegExp("gpt-5.4").test("gpt-5x4"));
    assert.ok(globToRegExp("o?").test("o3"));
    assert.ok(matchesModel(["*"], "anything", "p,anything"));
    // Patterns may target the qualified provider,model form.
    assert.ok(matchesModel(["deepseek,*"], "deepseek-chat", "deepseek,deepseek-chat"));
  }

  // ── model selection ───────────────────────────────────────────────
  {
    const all = selectModels(CONFIG, parseCodexConfigArgs([]));
    assert.deepEqual(
      all.map((m) => m.slug),
      [
        "deepseek,deepseek-chat",
        "deepseek,deepseek-reasoner",
        "openai,gpt-5.4",
        "openai,gpt-5-mini",
        "openai,o3",
      ]
    );
  }

  {
    const filtered = selectModels(
      CONFIG,
      parseCodexConfigArgs(["--providers", "openai", "--models", "gpt-*"])
    );
    assert.deepEqual(
      filtered.map((m) => m.slug),
      ["openai,gpt-5.4", "openai,gpt-5-mini"]
    );
  }

  {
    const none = selectModels(
      CONFIG,
      parseCodexConfigArgs(["--providers", "does-not-exist"])
    );
    assert.equal(none.length, 0);
  }

  // ── catalog entries ───────────────────────────────────────────────
  {
    const model: CatalogModel = {
      providerName: "deepseek",
      modelName: "deepseek-chat",
      slug: "deepseek,deepseek-chat",
      info: {
        key: "deepseek/deepseek-chat",
        provider: "deepseek",
        name: "DeepSeek Chat",
        context: 200_000,
        output: 8_192,
        reasoning: true,
        effortLevels: ["low", "medium", "high"],
        toolCall: true,
        attachment: true,
        modalitiesIn: ["text", "image"],
      },
    };
    const entry = buildCatalogEntry(model, 0, null);
    assert.equal(entry.slug, "deepseek,deepseek-chat");
    assert.equal(entry.display_name, "DeepSeek Chat");
    assert.equal(entry.context_window, 200_000);
    assert.equal(entry.max_context_window, 200_000);
    assert.equal(entry.auto_compact_token_limit, 160_000);
    // Codex's schema wants {effort, description} objects, and shell_type is
    // required or the whole catalog is rejected at startup.
    assert.equal(entry.shell_type, "shell_command");
    assert.deepEqual(
      entry.supported_reasoning_levels.map((l: any) => l.effort),
      ["low", "medium", "high"]
    );
    assert.equal(entry.supported_reasoning_levels[0].description, "Fast responses with lighter reasoning");
    assert.equal(entry.default_reasoning_level, "medium");
    assert.deepEqual(entry.input_modalities, ["text", "image"]);
    assert.equal(entry.visibility, "list");
    assert.equal(entry.supported_in_api, true);
    assert.ok(entry.comp_hash);
  }

  // Missing models.dev metadata uses the explicit Codex fallback contract.
  {
    const model: CatalogModel = {
      providerName: "custom",
      modelName: "mystery-model",
      slug: "custom,mystery-model",
      info: null,
    };
    const entry = buildCatalogEntry(model, 3, null);
    assert.equal(entry.display_name, "mystery-model");
    assert.equal(entry.context_window, 200_000);
    assert.equal(entry.shell_type, "shell_command");
    assert.deepEqual(
      entry.supported_reasoning_levels.map((l: any) => l.effort),
      ["low", "medium", "high"]
    );
    assert.equal(entry.default_reasoning_level, "medium");
    assert.deepEqual(entry.input_modalities, ["text"]);
    assert.equal(entry.priority, 103);
  }

  // A captured Codex template supplies fields we do not synthesize, and the
  // GPT-5 identity text is rewritten to the routed model's name.
  {
    const model: CatalogModel = {
      providerName: "p",
      modelName: "m",
      slug: "p,m",
      info: null,
    };
    const entry = buildCatalogEntry(model, 0, {
      base_instructions: "You are Codex, an agent based on GPT-5.4.",
      model_messages: {
        instructions_template: "You are a coding agent based on GPT-5.4-Codex.",
      },
    });
    assert.equal(entry.slug, "p,m");
    assert.equal(entry.shell_type, "shell_command");
    assert.ok(entry.base_instructions.includes("based on m"), "identity rewritten");
    assert.ok(!entry.base_instructions.includes("GPT-5"));
    assert.ok(!entry.base_instructions.includes(".4"));
    assert.ok(entry.model_messages.instructions_template.includes("based on m"));
    assert.ok(!entry.model_messages.instructions_template.includes("Codex."));
  }

  // ── config.toml patching ──────────────────────────────────────────
  const patchParams = {
    baseUrl: "http://127.0.0.1:3456/v1",
    catalogPath: "/home/u/.claude-code-router/codex/models.json",
    force: false,
    envKey: "CCR_API_KEY",
  };

  // Root keys must land before the first [table] header.
  {
    const original = ['model = "gpt-5.6"', "", "[mcp_servers.context7]", 'url = "https://x"'].join("\n");
    const { text, changed } = patchCodexConfig(original, patchParams);
    assert.equal(changed, true);

    const rootIdx = text.indexOf("model_provider");
    const tableIdx = text.indexOf("[mcp_servers.context7]");
    assert.ok(rootIdx !== -1 && tableIdx !== -1);
    assert.ok(rootIdx < tableIdx, "root keys must precede the first table");
    assert.ok(text.includes('model_provider = "ccr"'));
    assert.ok(text.includes('model_catalog_json = "/home/u/.claude-code-router/codex/models.json"'));

    assert.ok(text.includes("[model_providers.ccr]"));
    assert.ok(text.includes('base_url = "http://127.0.0.1:3456/v1"'));
    assert.ok(text.includes('env_key = "CCR_API_KEY"'));
    assert.ok(text.includes('wire_api = "responses"'));
    assert.ok(text.includes("requires_openai_auth = false"));
    assert.ok(!text.includes("openai_base_url"));
    // The user's own settings survive.
    assert.ok(text.includes('model = "gpt-5.6"'));
    assert.ok(text.includes('url = "https://x"'));
  }

  // Re-running replaces the managed blocks rather than stacking them.
  {
    const original = ['model = "gpt-5.6"', "", "[mcp_servers.x]", "enabled = false"].join("\n");
    const once = patchCodexConfig(original, patchParams).text;
    const twice = patchCodexConfig(once, patchParams);
    assert.equal(twice.changed, false, "second run must be a no-op");
    assert.equal(twice.text, once);
    assert.equal(twice.text.match(/# BEGIN ccr-managed/g)?.length, 1);
    assert.equal(twice.text.match(/\[model_providers\.ccr\]/g)?.length, 1);
  }

  // User-owned spacing outside the managed blocks is preserved.
  {
    const original = [
      'model = "gpt-5.6"',
      "",
      "",
      "",
      'model_reasoning_effort = "high"',
      "",
      "[mcp_servers.x]",
      "enabled = false",
    ].join("\n");
    const once = patchCodexConfig(original, patchParams).text;
    const twice = patchCodexConfig(once, patchParams);
    assert.ok(once.includes('model = "gpt-5.6"\n\n\n\nmodel_reasoning_effort'));
    assert.equal(twice.changed, false);
  }

  // Changing the base URL rewrites the managed block in place.
  {
    const once = patchCodexConfig("", patchParams).text;
    const moved = patchCodexConfig(once, {
      ...patchParams,
      baseUrl: "http://127.0.0.1:9999/v1",
    });
    assert.equal(moved.changed, true);
    assert.ok(moved.text.includes("http://127.0.0.1:9999/v1"));
    assert.ok(!moved.text.includes("http://127.0.0.1:3456/v1"));
    assert.equal(moved.text.match(/# BEGIN ccr-managed/g)?.length, 1);
  }

  // A user-owned key outside our block is never silently replaced.
  {
    const original = 'model_provider = "other"';
    assert.throws(
      () => patchCodexConfig(original, patchParams),
      /Refusing to replace user-owned model_provider/
    );

    const forced = patchCodexConfig(original, { ...patchParams, force: true });
    assert.ok(forced.text.includes('model_provider = "ccr"'));
    assert.ok(!forced.text.includes('model_provider = "other"'));
  }

  // An empty starting file is valid input.
  {
    const { text } = patchCodexConfig("", patchParams);
    assert.ok(text.startsWith("# BEGIN ccr-managed"));
    assert.ok(text.includes("[model_providers.ccr]"));
  }

  // Unauthenticated local CCR instances do not require a phantom env var.
  {
    const { text } = patchCodexConfig("", {
      ...patchParams,
      envKey: undefined,
    });
    assert.ok(!text.includes("env_key"));
  }

  // A damaged managed block must abort instead of deleting the rest of the file.
  {
    const malformed = [
      "# BEGIN ccr-managed",
      'model_provider = "ccr"',
      "[mcp_servers.keep_me]",
      'url = "https://example.test"',
    ].join("\n");
    assert.throws(
      () => patchCodexConfig(malformed, patchParams),
      /Malformed managed block/
    );
  }

  // A user-owned CCR provider table would make duplicate-table TOML invalid.
  {
    const original = [
      "[model_providers.ccr]",
      'name = "Existing CCR"',
      'base_url = "http://127.0.0.1:9999/v1"',
      "",
      "[mcp_servers.keep_me]",
      'url = "https://example.test"',
    ].join("\n");
    assert.throws(
      () => patchCodexConfig(original, patchParams),
      /Refusing to replace user-owned \[model_providers\.ccr\]/
    );

    const forced = patchCodexConfig(original, {
      ...patchParams,
      force: true,
    }).text;
    assert.equal(forced.match(/\[model_providers\.ccr\]/g)?.length, 1);
    assert.ok(!forced.includes("Existing CCR"));
    assert.ok(forced.includes("[mcp_servers.keep_me]"));
    assert.ok(forced.includes('url = "https://example.test"'));
  }

  console.log("codex-config: PASS");
}

try {
  main();
} catch (err) {
  console.error(err);
  process.exit(1);
}
