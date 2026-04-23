/**
 * Chat / Agent Factory model dropdown: use server-side catalog intersection when verified.
 * When catalog_verified is true but the list is empty, no models are selectable (do not fall back to raw enabled).
 */
export function getSelectableChatModels(enabledResponse) {
  if (!enabledResponse) return [];
  const {
    selectable_chat_models,
    catalog_verified,
    enabled_models,
    image_generation_model,
  } = enabledResponse;
  const img = image_generation_model || '';
  if (catalog_verified === true) {
    return Array.isArray(selectable_chat_models) ? selectable_chat_models : [];
  }
  return (enabled_models || []).filter((m) => m !== img);
}

/** True if catalog is loaded and modelId is in the chat-selectable list. */
export function isChatModelSelectable(enabledResponse, modelId) {
  if (!modelId) return false;
  const list = getSelectableChatModels(enabledResponse);
  return list.includes(modelId);
}

/**
 * If modelId is in the selectable list, return it; otherwise first selectable.
 * If catalog missing or selectable list empty, returns modelId unchanged (caller should not POST select yet).
 */
export function coerceChatModelToSelectable(enabledResponse, modelId) {
  const list = getSelectableChatModels(enabledResponse);
  if (!enabledResponse || list.length === 0) return modelId || '';
  if (modelId && list.includes(modelId)) return modelId;
  return list[0];
}
