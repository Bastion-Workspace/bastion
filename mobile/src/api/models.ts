import { apiRequest } from './client';

export type ModelRoles = {
  user_chat_model?: string;
  user_fast_model?: string;
  user_image_gen_model?: string;
  user_image_analysis_model?: string;
};

export async function getModelRoles(): Promise<ModelRoles> {
  return apiRequest<ModelRoles>('/api/user/models/roles');
}

export type EnabledModel = {
  model_id: string;
  display_name: string;
  provider_id: number;
};

export async function getEnabledModels(): Promise<EnabledModel[]> {
  const res = await apiRequest<{ enabled_models?: EnabledModel[] }>('/api/user/models/enabled');
  return res.enabled_models ?? [];
}

export async function setUserChatModelRole(modelId: string): Promise<void> {
  await apiRequest<{ status?: string }>('/api/user/models/roles', {
    method: 'PUT',
    body: JSON.stringify({ user_chat_model: modelId }),
  });
}
