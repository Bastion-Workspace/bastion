export type PushTokenBody = {
  token: string;
  platform: string;
  device_id: string;
  app_version?: string;
};

export async function registerPushToken(
  baseUrl: string,
  bearer: string,
  body: PushTokenBody
): Promise<void> {
  const res = await fetch(`${baseUrl}/api/notifications/push-token`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${bearer}`,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const t = await res.text();
    throw new Error(`push-token failed: ${res.status} ${t}`);
  }
}

export async function revokePushToken(
  baseUrl: string,
  bearer: string,
  deviceId: string
): Promise<void> {
  const enc = encodeURIComponent(deviceId);
  const res = await fetch(`${baseUrl}/api/notifications/push-token/${enc}`, {
    method: 'DELETE',
    headers: {
      Authorization: `Bearer ${bearer}`,
    },
  });
  if (!res.ok && res.status !== 404) {
    const t = await res.text();
    throw new Error(`push-token revoke failed: ${res.status} ${t}`);
  }
}
