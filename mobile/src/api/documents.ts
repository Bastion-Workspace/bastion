import { apiRequest } from './client';

export type DocumentInfo = {
  document_id: string;
  filename: string;
  title?: string | null;
  collection_type?: string;
};

export type DocumentListResponse = {
  documents: DocumentInfo[];
  total: number;
};

export async function listUserDocuments(skip = 0, limit = 100): Promise<DocumentListResponse> {
  return apiRequest<DocumentListResponse>(`/api/user/documents?skip=${skip}&limit=${limit}`);
}

export type DocumentContentResponse = {
  content?: string;
  is_encrypted?: boolean;
  requires_password?: boolean;
  metadata?: Record<string, unknown>;
  total_length?: number;
};

export async function getDocumentContent(documentId: string): Promise<DocumentContentResponse> {
  return apiRequest<DocumentContentResponse>(`/api/documents/${documentId}/content`);
}

export type DocumentSearchResultRow = {
  document_id?: string;
  similarity_score?: number;
  text?: string;
  context?: { text?: string };
  document?: {
    document_id?: string;
    filename?: string;
    title?: string | null;
  };
};

export type DocumentSearchResponse = {
  results: DocumentSearchResultRow[];
  total_results?: number;
  search_mode?: string;
};

export async function searchDocuments(query: string, limit = 30): Promise<DocumentSearchResponse> {
  const q = query.trim();
  if (!q) {
    return { results: [], total_results: 0 };
  }
  return apiRequest<DocumentSearchResponse>('/api/user/documents/search', {
    method: 'POST',
    body: JSON.stringify({
      query: q,
      search_mode: 'hybrid',
      limit,
    }),
  });
}
