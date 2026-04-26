import { apiRequest } from './client';

export type ConversationSummary = {
  conversation_id: string;
  title?: string | null;
  last_message_at?: string | null;
  message_count?: number;
};

export type ConversationListResponse = {
  conversations: ConversationSummary[];
  total_count: number;
  has_more: boolean;
};

export async function listConversations(skip = 0, limit = 50): Promise<ConversationListResponse> {
  return apiRequest<ConversationListResponse>(
    `/api/conversations?skip=${skip}&limit=${limit}`
  );
}

export type ConversationResponse = {
  conversation: ConversationSummary & Record<string, unknown>;
};

export async function createConversation(body: {
  title?: string | null;
  initial_message?: string | null;
}): Promise<ConversationResponse> {
  return apiRequest<ConversationResponse>('/api/conversations', {
    method: 'POST',
    body: JSON.stringify({
      title: body.title ?? null,
      description: null,
      tags: [],
      folder_id: null,
      initial_message: body.initial_message ?? null,
    }),
  });
}

export type ConversationMessage = {
  message_id: string;
  message_type: string;
  content: string;
  created_at: string;
};

export type MessageListResponse = {
  messages: ConversationMessage[];
  total_count: number;
  has_more: boolean;
};

export async function getConversationMessages(
  conversationId: string,
  skip = 0,
  limit = 100
): Promise<MessageListResponse> {
  return apiRequest<MessageListResponse>(
    `/api/conversations/${conversationId}/messages?skip=${skip}&limit=${limit}`
  );
}

export async function addUserMessage(conversationId: string, content: string): Promise<unknown> {
  return apiRequest(`/api/conversations/${conversationId}/messages`, {
    method: 'POST',
    body: JSON.stringify({
      content,
      message_type: 'user',
      metadata: {},
    }),
  });
}
