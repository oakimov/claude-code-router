import {
  copyCurlCommand,
  parseHeadersJson,
} from "../lib/debugChat";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function testLineHeaders(): void {
  const parsed = parseHeadersJson("X-Test: one\nAuthorization: Bearer secret");
  assert(
    parsed["X-Test"] === "one" &&
      parsed.Authorization === "Bearer secret",
    "line-based headers should be parsed"
  );
}

function testCurlUsesPosixSafeQuoting(): void {
  const command = copyCurlCommand({
    url: "https://example.test/'$(touch unsafe)",
    method: "POST",
    headers: {
      "X-Test": "`echo unsafe` $HOME 'quoted'",
      Authorization: "Bearer secret",
    },
    body: { prompt: "$(echo unsafe)\nsecond line", quote: "it's safe" },
  });

  assert(command.includes("Bearer PLACEHOLDER"), "auth values should be redacted");
  assert(!command.includes("Bearer secret"), "auth secrets should not be copied");
  assert(command.includes("'\\''"), "single quotes should use POSIX escaping");
  assert(
    command.includes("--data-raw '{") &&
      command.includes('"prompt": "$(echo unsafe)'),
    "command substitutions should remain inside single quotes"
  );
}

testLineHeaders();
testCurlUsesPosixSafeQuoting();
console.log("debug-chat ui helpers: PASS");
