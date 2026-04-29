/**
 * Resolves `import { Audio } from 'expo-av'` before `npm install` pulls the real package.
 */
declare module 'expo-av' {
  export namespace Audio {
    function requestPermissionsAsync(): Promise<{ granted: boolean; status?: string }>;
    function setAudioModeAsync(partial: Record<string, unknown>): Promise<void>;
    const RecordingOptionsPresets: { HIGH_QUALITY: unknown };

    class Recording {
      static createAsync(options: unknown): Promise<{
        recording: { stopAndUnloadAsync(): Promise<void>; getURI(): string | null };
      }>;
    }
  }
}
