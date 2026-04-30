import type { MusicTrack } from '../api/media';

let cachedTracks: MusicTrack[] = [];

export function setMediaSearchTracksSeed(tracks: MusicTrack[]): void {
  cachedTracks = tracks.slice();
}

export function consumeMediaSearchTracksSeed(): MusicTrack[] {
  const out = cachedTracks;
  cachedTracks = [];
  return out;
}
