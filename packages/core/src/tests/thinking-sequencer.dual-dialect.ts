/**
 * Unit tests for ThinkingSequencer dual-dialect support:
 * public Gemini API vs Antigravity content-then-signature streams.
 */
import { ThinkingSequencer } from "../utils/google.util";

type Event =
  | { type: "thinking"; content: string }
  | { type: "signature"; sig: string }
  | { type: "content"; text: string; mode?: string };

function collect(): { events: Event[]; sequencer: ThinkingSequencer } {
  const events: Event[] = [];
  const sequencer = new ThinkingSequencer({
    thinking: (content) => events.push({ type: "thinking", content }),
    signature: (sig) => events.push({ type: "signature", sig }),
    content: (text, meta) =>
      events.push({ type: "content", text, mode: meta?.mode }),
  });
  return { events, sequencer };
}

function assert(cond: unknown, msg: string): void {
  if (!cond) throw new Error(msg);
}

function run() {
  // Public Gemini happy path: think → sig → content
  {
    const { events, sequencer } = collect();
    sequencer.processThinking("Let me think");
    sequencer.processSignature("sig_a");
    sequencer.processContent("Answer");
    assert(
      events.map((e) => e.type).join(",") === "thinking,signature,content",
      `happy-path order: ${events.map((e) => e.type).join(",")}`
    );
  }

  // Public Gemini 3 out-of-order: think → buffer content → sig flushes
  {
    const { events, sequencer } = collect();
    sequencer.processThinking("Thinking...");
    assert(
      sequencer.shouldDeferContent(false, false) === true,
      "should defer after thinking"
    );
    sequencer.bufferContent("Final answer text");
    sequencer.processSignatureWithMeta("sig_delayed", undefined, {
      flushMeta: { mode: "buffered" },
    });
    assert(events[0]?.type === "thinking", "thinking first");
    assert(events[1]?.type === "signature", "signature second");
    assert(
      events[2]?.type === "content" &&
        (events[2] as any).text === "Final answer text",
      "flushed buffered content after signature"
    );
  }

  // Empty thinking: signature first emits placeholder thinking
  {
    const { events, sequencer } = collect();
    sequencer.processSignature("sig_empty");
    sequencer.processContent("Content after empty thinking");
    assert(
      events[0]?.type === "thinking" &&
        (events[0] as any).content === "(no content)",
      "placeholder thinking on early signature"
    );
    assert(events[1]?.type === "signature", "signature after placeholder");
    assert(events[2]?.type === "content", "content after signature");
  }

  // Antigravity: content then signature trailer — no late thinking
  {
    const { events, sequencer } = collect();
    sequencer.processContent("Hello from Antigravity");
    sequencer.processSignature("sig_antigravity_trailer");
    assert(events.length === 1, `expected only content, got ${events.length}`);
    assert(
      events[0]?.type === "content" &&
        (events[0] as any).text === "Hello from Antigravity",
      "antigravity content preserved without late thinking"
    );
    assert(sequencer.signatureSent === true, "signature marked handled");
    assert(sequencer.contentSent === true, "content marked sent");
  }

  // Antigravity same-chunk simulation: content then sig in sequence
  {
    const { events, sequencer } = collect();
    sequencer.processContent("One-shot reply");
    sequencer.processSignature("sig_same_chunk");
    assert(
      !events.some((e) => e.type === "thinking"),
      "no thinking after content-first same-chunk"
    );
    assert(
      events.some((e) => e.type === "content"),
      "content still emitted"
    );
  }

  console.log("ThinkingSequencer dual-dialect tests: PASS");
}

run();
