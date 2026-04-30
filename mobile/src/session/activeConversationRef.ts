let activeConversationId: string | null = null;

export function setActiveConversationForNotifications(id: string | null): void {
  activeConversationId = id && String(id).trim() ? String(id).trim() : null;
}

export function getActiveConversationForNotifications(): string | null {
  return activeConversationId;
}
