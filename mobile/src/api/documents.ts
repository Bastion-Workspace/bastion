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

export async function listUserDocuments(offset = 0, limit = 100): Promise<DocumentListResponse> {
  return apiRequest<DocumentListResponse>(
    `/api/user/documents?offset=${offset}&limit=${limit}`
  );
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
