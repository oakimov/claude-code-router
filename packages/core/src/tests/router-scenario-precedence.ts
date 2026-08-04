/**
 * Routing precedence (Fix 7): explicit provider,model → subagent →
 * background → webSearch → think → longContext → default. A request that is
 * both long and matches a higher-priority scenario keeps the scenario route.
 */
import assert from "node:assert/strict";
import { router } from "../utils/router";

const Router = {
  default: "d,m",
  background: "b,m",
  think: "t,m",
  longContext: "l,m",
  webSearch: "w,m",
  // Tiny threshold so ordinary test messages count as "long context".
  longContextThreshold: 1,
};

const providers = [
  { name: "p", models: ["m"] },
  { name: "d", models: ["m"] },
];

const configService = {
  get(key: string) {
    if (key === "providers") return providers;
    if (key === "Router") return Router;
    return undefined;
  },
  getAll() {
    return { Router, providers };
  },
} as any;

function makeReq(body: any, extra: Record<string, any> = {}): any {
  return {
    body,
    log: { info() {}, warn() {}, error() {}, debug() {} },
    ...extra,
  };
}

async function route(body: any, extra: Record<string, any> = {}) {
  const req = makeReq(body, extra);
  await router(req, null, { configService });
  return { model: req.body.model, scenarioType: req.scenarioType };
}

async function main() {
  const longMessage = { role: "user", content: "a reasonably sized prompt" };

  // Explicit provider,model beats everything, including long context.
  {
    const result = await route({ model: "p,m", messages: [longMessage] });
    assert.equal(result.model, "p,m");
    assert.equal(result.scenarioType, "default");
  }

  // Subagent tag beats long context.
  {
    const result = await route(
      { model: "claude-sonnet", messages: [longMessage] },
      { protocolContext: { taggedSubagentModel: "s,m" } }
    );
    assert.equal(result.model, "s,m");
    assert.equal(result.scenarioType, "subagent");
  }

  // Background (haiku) beats long context.
  {
    const result = await route({
      model: "claude-haiku-4-5",
      messages: [longMessage],
    });
    assert.equal(result.model, "b,m");
    assert.equal(result.scenarioType, "background");
  }

  // Web search beats think and long context.
  {
    const result = await route({
      model: "claude-sonnet",
      messages: [longMessage],
      tools: [{ type: "web_search" }],
      thinking: { type: "enabled", budget_tokens: 1024 },
    });
    assert.equal(result.model, "w,m");
    assert.equal(result.scenarioType, "webSearch");
  }

  // Think beats long context.
  {
    const result = await route({
      model: "claude-sonnet",
      messages: [longMessage],
      thinking: { type: "enabled", budget_tokens: 1024 },
    });
    assert.equal(result.model, "t,m");
    assert.equal(result.scenarioType, "think");
  }

  // Long-context-only request still routes to the long context model.
  {
    const result = await route({
      model: "claude-sonnet",
      messages: [longMessage],
    });
    assert.equal(result.model, "l,m");
    assert.equal(result.scenarioType, "longContext");
  }

  // Short, scenario-free request falls to default.
  {
    const result = await route({ model: "claude-sonnet", messages: [] });
    assert.equal(result.model, "d,m");
    assert.equal(result.scenarioType, "default");
  }

  console.log("router-scenario-precedence: all tests passed");
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
