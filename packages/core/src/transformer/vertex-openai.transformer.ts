import { GoogleAuth } from "google-auth-library";
import { UnifiedChatRequest } from "../types/llm";
import { Transformer, TransformerOptions } from "../types/transformer";
import {
  stripMessagesCacheControl,
  stripToolsCacheControl,
} from "../utils/cacheControl";

export interface VertexOpenaiOptions extends TransformerOptions {
  client_email?: string;
  private_key?: string;
}

/**
 * Auth-only transformer for Vertex AI OpenAI-compatible endpoints.
 * Injects a Google service-account Bearer token into outbound requests.
 */
export class VertexOpenaiTransformer implements Transformer {
  static TransformerName = "vertex-openai";

  private client?: Awaited<ReturnType<GoogleAuth["getClient"]>>;
  private readonly client_email?: string;
  private readonly private_key?: string;

  constructor(options: VertexOpenaiOptions = {}) {
    this.client_email = options.client_email;
    this.private_key = options.private_key?.replace(/\\n/g, "\n");
  }

  private async getClient() {
    if (this.client) {
      return this.client;
    }

    if (!this.client_email || !this.private_key) {
      throw new Error(
        "vertex-openai requires client_email and private_key in transformer options"
      );
    }

    const auth = new GoogleAuth({
      scopes: ["https://www.googleapis.com/auth/cloud-platform"],
      credentials: {
        client_email: this.client_email,
        private_key: this.private_key,
      },
    });

    this.client = await auth.getClient();
    return this.client;
  }

  async transformRequestIn(request: UnifiedChatRequest): Promise<Record<string, any>> {
    const client = await this.getClient();
    const { token } = await client.getAccessToken();

    if (!token) {
      throw new Error("Failed to get access token");
    }

    return {
      body: {
        ...request,
        messages: stripMessagesCacheControl(request.messages),
        tools: stripToolsCacheControl(request.tools),
      },
      config: {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      },
    };
  }
}
