/** Help topic id (path under backend/help_docs, without .md). */
export const HELP_TOPIC_DOCUMENT_ENCRYPTION =
  'documents-and-folders/10-document-encryption';

/**
 * Open the in-app Help panel to a topic (Navigation listens for this event).
 * @param {string} topicId
 */
export function openHelpTopic(topicId) {
  if (typeof window === 'undefined' || !topicId) return;
  window.dispatchEvent(
    new CustomEvent('bastion-open-help', { detail: { topicId } })
  );
}
