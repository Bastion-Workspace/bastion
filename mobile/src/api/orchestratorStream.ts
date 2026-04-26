import { assertApiBaseUrl } from './config';
import { getStoredToken } from '../session/tokenStore';

export type StreamChunk =
  | { type: 'content'; content: string }
  | { type: 'complete'; content?: string }
  | { type: 'run_started'; run_id?: string }
  | { type: 'cancelled' }
  | { type: 'title'; message?: string }
  | { type: 'status'; message?: string; content?: string }
  | { type: string; [key: string]: unknown };

export type StreamOrchestratorParams = {
  query: string;
  conversation_id: string;
  session_id?: string;
  user_chat_model?: string;
  agent_profile_id?: string;
  signal?: AbortSignal;
  onChunk: (chunk: StreamChunk) => void;
};

/**
 * POST /api/async/orchestrator/stream with SSE-style `data: {...}` lines (fetch + ReadableStream).
 */
export async function streamOrchestrator({
  query,
  conversation_id,
  session_id = 'bastion-mobile',
  user_chat_model,
  agent_profile_id,
  signal,
  onChunk,
}: StreamOrchestratorParams): Promise<void> {
  const base = assertApiBaseUrl();
  const token = await getStoredToken();
  if (!token) {
    throw new Error('Not authenticated');
  }

  const body = {
    query,
    conversation_id,
    session_id,
    persist_conversation: true,
    user_chat_model: user_chat_model || undefined,
    agent_profile_id: agent_profile_id || undefined,
  };

  const res = await fetch(`${base}/api/async/orchestrator/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok) {
    const t = await res.text();
    throw new Error(`Stream failed: ${res.status} ${t}`);
  }

  const reader = res.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() ?? '';
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const raw = line.slice(6).trim();
      if (!raw) continue;
      try {
        const data = JSON.parse(raw) as StreamChunk;
        onChunk(data);
      } catch {
        /* skip malformed line */
      }
    }
  }
  if (buffer.startsWith('data: ')) {
    try {
      onChunk(JSON.parse(buffer.slice(6).trim()) as StreamChunk);
    } catch {
      /* ignore */
    }
  }
}
