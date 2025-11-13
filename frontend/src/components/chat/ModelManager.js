import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../../services/apiService';

export const useModelManager = () => {
  const queryClient = useQueryClient();

  // Fetch enabled models for dropdown
  const { data: enabledModelsData } = useQuery(
    'enabledModels',
    () => apiService.getEnabledModels(),
    {
      refetchInterval: 30000,
    }
  );

  // Fetch current model
  const { data: currentModelData } = useQuery(
    'currentModel',
    () => apiService.getCurrentModel(),
    {
      refetchInterval: 10000,
    }
  );

  // Fetch available models for display names
  const { data: availableModelsData } = useQuery(
    'availableModels',
    () => apiService.getAvailableModels(),
    {
      staleTime: 300000,
    }
  );

  // Model selection mutation
  const selectModelMutation = useMutation(
    (modelName) => apiService.selectModel(modelName),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('currentModel');
      },
      onError: (error) => {
        console.error('Failed to select model:', error);
      },
    }
  );

  const handleModelSelect = (modelName) => {
    selectModelMutation.mutate(modelName);
  };

  return {
    enabledModelsData,
    currentModelData,
    availableModelsData,
    selectModelMutation,
    handleModelSelect,
  };
}; 