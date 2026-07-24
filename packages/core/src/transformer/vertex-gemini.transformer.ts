import { LLMProvider, UnifiedChatRequest } from "../types/llm";
import { Transformer } from "../types/transformer";
import {
  buildRequestBody,
  transformRequestOut,
  transformResponseOut,
} from "../utils/gemini.util";
import { attachGeminiCachedContent } from "../utils/gemini-cache";

async function getAccessToken(logger?: any): Promise<string> {
  try {
    const { GoogleAuth } = await import('google-auth-library');

    const auth = new GoogleAuth({
      scopes: ['https://www.googleapis.com/auth/cloud-platform']
    });

    const client = await auth.getClient();
    const accessToken = await client.getAccessToken();
    return accessToken.token || '';
  } catch (error) {
    (logger?.error ?? console.error)('Error getting access token:', error);
    throw new Error('Failed to get access token for Vertex AI. Please ensure you have set up authentication using one of these methods:\n' +
      '1. Set GOOGLE_APPLICATION_CREDENTIALS to point to service account key file\n' +
      '2. Run "gcloud auth application-default login"\n' +
      '3. Use Google Cloud environment with default service account');
  }
}

export class VertexGeminiTransformer implements Transformer {
  logger?: any;
  name = "vertex-gemini";

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider,
    context?: any
  ): Promise<Record<string, any>> {
    let projectId = process.env.GOOGLE_CLOUD_PROJECT;
    const location = process.env.GOOGLE_CLOUD_LOCATION || 'us-central1';

    if (!projectId && process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      try {
        const fs = await import('fs');
        const keyContent = fs.readFileSync(process.env.GOOGLE_APPLICATION_CREDENTIALS, 'utf8');
        const credentials = JSON.parse(keyContent);
        if (credentials && credentials.project_id) {
          projectId = credentials.project_id;
        }
      } catch (error) {
        this.logger?.error('Error extracting project_id from GOOGLE_APPLICATION_CREDENTIALS:', error);
      }
    }

    if (!projectId) {
      throw new Error('Project ID is required for Vertex AI. Set GOOGLE_CLOUD_PROJECT environment variable or ensure project_id is in GOOGLE_APPLICATION_CREDENTIALS file.');
    }

    const accessToken = await getAccessToken(this.logger);
    const model = request.model || provider.model || "";
    const baseUrl =
      provider.baseUrl.endsWith('/') ? provider.baseUrl : provider.baseUrl + '/' || `https://${location}-aiplatform.googleapis.com`;
    const modelResource = `projects/${projectId}/locations/${location}/publishers/google/models/${model}`;
    const body = await attachGeminiCachedContent({
      body: buildRequestBody(request),
      modelResource,
      createUrl: new URL(
        `./v1beta1/projects/${projectId}/locations/${location}/cachedContents`,
        baseUrl
      ),
      headers: {
        Authorization: `Bearer ${accessToken}`,
        "x-goog-api-key": undefined,
      },
      logger: this.logger,
    });
    return {
      body,
      config: {
        url: new URL(
          `./v1beta1/${modelResource}:${request.stream ? "streamGenerateContent" : "generateContent"}`,
          baseUrl
        ),
        headers: {
          "Authorization": `Bearer ${accessToken}`,
          "x-goog-api-key": undefined,
        },
      },
    };
  }

  async transformRequestOut(request: Record<string, any>): Promise<UnifiedChatRequest> {
    return transformRequestOut(request);
  }

  async transformResponseOut(response: Response): Promise<Response> {
    return transformResponseOut(response, this.name);
  }
}
