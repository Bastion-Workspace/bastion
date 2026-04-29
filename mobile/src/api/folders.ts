import { apiRequest } from './client';

export type DocumentFolderNode = {
  folder_id: string;
  name: string;
  parent_folder_id?: string | null;
  collection_type?: string;
  document_count?: number | null;
  subfolder_count?: number | null;
  children?: DocumentFolderNode[];
};

export type FolderTreeApiResponse = {
  folders?: DocumentFolderNode[];
  total_folders?: number;
};

export async function getFolderTree(): Promise<FolderTreeApiResponse> {
  return apiRequest<FolderTreeApiResponse>('/api/folders/tree');
}

export type FolderDocumentRow = {
  document_id: string;
  filename: string;
  title?: string | null;
};

export type FolderContentsApiResponse = {
  folder?: { folder_id: string; name: string };
  documents?: FolderDocumentRow[];
  subfolders?: DocumentFolderNode[];
  total_documents?: number;
  total_subfolders?: number;
};

export async function getFolderContents(
  folderId: string,
  limit = 100,
  offset = 0
): Promise<FolderContentsApiResponse> {
  const p = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return apiRequest<FolderContentsApiResponse>(
    `/api/folders/${encodeURIComponent(folderId)}/contents?${p.toString()}`
  );
}
