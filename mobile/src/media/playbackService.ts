import TrackPlayer, { Event } from 'react-native-track-player';

/**
 * Headset / lock-screen / notification remote events (runs in playback service).
 */
export default async function PlaybackService(): Promise<void> {
  TrackPlayer.addEventListener(Event.RemotePlay, () => {
    void TrackPlayer.play();
  });
  TrackPlayer.addEventListener(Event.RemotePause, () => {
    void TrackPlayer.pause();
  });
  TrackPlayer.addEventListener(Event.RemoteNext, () => {
    void TrackPlayer.skipToNext();
  });
  TrackPlayer.addEventListener(Event.RemotePrevious, () => {
    void TrackPlayer.skipToPrevious();
  });
  TrackPlayer.addEventListener(Event.RemoteSeek, (event: { position: number }) => {
    void TrackPlayer.seekTo(event.position);
  });
  TrackPlayer.addEventListener(Event.RemoteStop, () => {
    void TrackPlayer.stop();
  });
}
