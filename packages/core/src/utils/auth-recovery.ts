export type UnauthorizedAuthRecovery = () => Promise<
  Record<string, string> | null
>;

export async function sendWithUnauthorizedAuthRecovery(
  send: (headers: Record<string, string>) => Promise<Response>,
  requestHeaders: Record<string, string>,
  recover?: UnauthorizedAuthRecovery
): Promise<Response> {
  let response = await send(requestHeaders);
  if (response.status !== 401 || typeof recover !== "function") {
    return response;
  }

  const recoveredHeaders = await recover();
  if (!recoveredHeaders) return response;

  try {
    await response.body?.cancel();
  } catch {
    // The unauthorized response may already be fully consumed.
  }
  Object.assign(requestHeaders, recoveredHeaders);
  return send(requestHeaders);
}
