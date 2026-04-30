/**
 * Minimal typings when `expo-file-system` is not yet installed in the workspace.
 * Remove or narrow if you rely on the full package types from node_modules.
 */
declare module 'expo-file-system' {
  export enum EncodingType {
    UTF8 = 'utf8',
    Base64 = 'base64',
  }

  export const documentDirectory: string | null;
  export const cacheDirectory: string | null;

  export function getInfoAsync(
    uri: string,
    options?: { size?: boolean; md5?: boolean }
  ): Promise<{ exists: boolean; isDirectory?: boolean; uri?: string; size?: number }>;

  export function makeDirectoryAsync(uri: string, options?: { intermediates?: boolean }): Promise<void>;

  export function writeAsStringAsync(
    uri: string,
    contents: string,
    options?: { encoding?: EncodingType }
  ): Promise<void>;

  export function readAsStringAsync(uri: string, options?: { encoding?: EncodingType }): Promise<string>;

  export function deleteAsync(uri: string, options?: { idempotent?: boolean }): Promise<void>;

  export function readDirectoryAsync(uri: string): Promise<string[]>;

  export function copyAsync(options: { from: string; to: string }): Promise<void>;
}
