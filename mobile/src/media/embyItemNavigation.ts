import { Alert } from 'react-native';
import type { EmbyItem } from '../api/emby';

const OPEN_AS_ALBUM = new Set(['MusicAlbum', 'Playlist', 'Folder', 'BoxSet']);

type RouterPush = {
  push: (href: { pathname: string; params?: Record<string, string | undefined> }) => void;
};

/**
 * Shared Emby item taps: video, series, music library, and alerts for unsupported types.
 */
export function navigateFromEmbyItem(item: EmbyItem, router: RouterPush): void {
  const t = item.Type ?? '';
  if (t === 'Movie' || t === 'Episode' || t === 'Video' || t === 'MusicVideo') {
    router.push({
      pathname: '/(app)/media/video',
      params: {
        itemId: item.Id,
        title: item.Name ?? 'Video',
        startTimeTicks: String(item.UserData?.PlaybackPositionTicks ?? 0),
      },
    });
    return;
  }
  if (t === 'Series') {
    router.push({
      pathname: '/(app)/media/emby-series',
      params: {
        seriesId: item.Id,
        seriesName: item.Name ?? 'Series',
      },
    });
    return;
  }
  if (t === 'MusicArtist') {
    router.push({
      pathname: '/(app)/media/[parentId]',
      params: {
        parentId: item.Id,
        type: 'artist',
        title: item.Name ?? 'Artist',
        serviceType: 'emby',
      },
    });
    return;
  }
  let parentId = item.Id;
  let navType: 'album' | 'playlist' = 'album';
  if (t === 'Audio') {
    const albumOrParent = item.AlbumId || item.ParentId;
    if (albumOrParent) {
      parentId = albumOrParent;
    }
  } else if (t === 'Playlist') {
    navType = 'playlist';
  } else if (!OPEN_AS_ALBUM.has(t) && t !== 'MusicAlbum') {
    Alert.alert('Not available', 'This item type cannot be opened in the mobile music player.');
    return;
  }
  router.push({
    pathname: '/(app)/media/[parentId]',
    params: {
      parentId,
      type: navType,
      title: item.Name ?? 'Media',
      serviceType: 'emby',
    },
  });
}
