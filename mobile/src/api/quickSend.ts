import { createConversation } from './conversations';
import { getEnabledModels, getModelRoles } from './models';
import { streamOrchestrator } from './orchestratorStream';

/** Practical URL / Shortcut length limit for quick-send payloads. */
export const QUICK_SEND_MAX_MESSAGE_CHARS = 2000;

export type QuickSendOptions = {
  /** Conversation title in Bastion (default: Quick send). */
  title?: string;
  /** Orchestrator session_id for analytics (default: bastion-mobile-shortcut). */
  sessionId?: string;
};

async function resolveUserChatModel(): Promise<string> {
  try {
    const [roles, models] = await Promise.all([getModelRoles(), getEnabledModels()]);
    const role = (roles.user_chat_model ?? '').trim();
    if (role) return role;
    if (models.length > 0) return models[0].model_id;
  } catch {
    /* fall through */
  }
  return '';
}

/**
 * Create a conversation with the user message and fire-and-forget the orchestrator stream.
 * Used by voice capture (after STT) and by deep-link shortcut-send.
 */
export async function quickSendToDefaultAgent(
  text: string,
  opts: QuickSendOptions = {}
): Promise<{ conversation_id: string }> {
  const trimmed = text.trim();
  if (!trimmed) {
    throw new Error('Empty message');
  }
  if (trimmed.length > QUICK_SEND_MAX_MESSAGE_CHARS) {
    throw new Error(`Message too long (max ${QUICK_SEND_MAX_MESSAGE_CHARS} characters)`);
  }

  const userChatModel = (await resolveUserChatModel()).trim();

  const res = await createConversation({
    title: opts.title ?? 'Quick send',
    initial_message: trimmed,
  });
  const cid = res.conversation?.conversation_id ?? null;
  if (!cid) {
    throw new Error('Could not create conversation');
  }

  const sessionId = opts.sessionId ?? 'bastion-mobile-shortcut';

  void streamOrchestrator({
    query: trimmed,
    conversation_id: cid,
    session_id: sessionId,
    user_chat_model: userChatModel || undefined,
    onChunk: () => {},
  }).catch(() => {
    /* Fire-and-forget; errors surface in Chat when user opens the thread */
  });

  return { conversation_id: cid };
}
