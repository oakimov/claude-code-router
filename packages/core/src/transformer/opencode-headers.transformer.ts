import { randomBytes, createHash } from "crypto";
import { Transformer } from "@/types/transformer";

const BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

export class OpencodeHeadersTransformer implements Transformer {
  name = "opencode-headers";

  private sessionCache = new Map<string, string>();
  private readonly MAX_SESSIONS = 100;
  private lastTimestamp = 0;
  private counter = 0;

  async transformRequestIn(
    request: any,
    provider: any,
    context: any
  ): Promise<Record<string, any>> {
    const conversationId = context?.req?.sessionId || this.fingerprintConversation(request, context);
    const sessionId = this.getOrCreateSessionId(conversationId);
    const requestId = this.generateId("msg");

    return {
      body: request.body || request,
      config: {
        ...request.config,
        headers: {
          ...request.config?.headers,
          "x-api-key": provider.apiKey || "",
          "x-opencode-project": "global",
          "x-opencode-session": sessionId,
          "x-opencode-request": requestId,
          "x-opencode-client": "cli",
          "user-agent": "opencode/1.17.1",
          authorization: undefined,
        },
      },
    };
  }

  private fingerprintConversation(request: any, context: any): string {
    const body = request.body || request;
    const model = body.model || "";
    const msgs = body.messages || [];
    const sample = JSON.stringify(msgs.slice(0, 3));
    const ip = context?.req?.headers?.["x-forwarded-for"] || context?.req?.ip || "";
    const ua = context?.req?.headers?.["user-agent"] || "";
    return createHash("sha256")
      .update(`${model}|${ip}|${ua}|${sample}`)
      .digest("hex")
      .slice(0, 32);
  }

  private getOrCreateSessionId(key: string): string {
    const existing = this.sessionCache.get(key);
    if (existing) return existing;

    if (this.sessionCache.size >= this.MAX_SESSIONS) {
      const oldest = this.sessionCache.keys().next().value;
      if (oldest !== undefined) this.sessionCache.delete(oldest);
    }

    const id = this.generateId("ses");
    this.sessionCache.set(key, id);
    return id;
  }

  private generateId(prefix: string): string {
    const now = Date.now();
    if (now !== this.lastTimestamp) {
      this.lastTimestamp = now;
      this.counter = 0;
    }
    this.counter++;

    const ts = BigInt(now) * BigInt(0x1000) + BigInt(this.counter);
    const timeBytes = Buffer.alloc(6);
    for (let i = 0; i < 6; i++) {
      timeBytes[i] = Number((ts >> BigInt(40 - 8 * i)) & BigInt(0xff));
    }

    const random = randomBytes(14);
    let suffix = "";
    for (let i = 0; i < 14; i++) suffix += BASE62[random[i] % 62];

    return `${prefix}_${timeBytes.toString("hex")}${suffix}`;
  }
}
