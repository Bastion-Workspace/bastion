import { Image, Pressable, StyleSheet, Text, View } from 'react-native';
import type { OpdsFeedEntry } from '../../api/ebooks';
import { pickCoverHrefFromEntry } from './opdsUtils';
import { getColors, type AppColors } from '../../theme/colors';

type Props = {
  entry: OpdsFeedEntry;
  baseUrl: string;
  scheme: 'light' | 'dark' | null | undefined;
  acquisitionFormat?: 'epub' | 'pdf';
  onPressAcquisition?: () => void;
  onPressNavigation?: () => void;
  disabled?: boolean;
};

export function BookCard({
  entry,
  baseUrl,
  scheme,
  acquisitionFormat,
  onPressAcquisition,
  onPressNavigation,
  disabled,
}: Props) {
  const c: AppColors = getColors(scheme === 'dark' ? 'dark' : 'light');
  const title = (entry.title || 'Untitled').trim();
  const cover = pickCoverHrefFromEntry(entry, baseUrl);
  const hasAcq = Boolean(onPressAcquisition);
  const hasNav = Boolean(onPressNavigation);

  return (
    <Pressable
      style={[styles.row, { borderColor: c.border, backgroundColor: c.surface }]}
      onPress={() => {
        if (hasAcq) onPressAcquisition?.();
        else if (hasNav) onPressNavigation?.();
      }}
      disabled={disabled || (!hasAcq && !hasNav)}
      accessibilityRole="button"
      accessibilityLabel={title}
    >
      {cover ? (
        <Image source={{ uri: cover }} style={styles.thumb} resizeMode="cover" />
      ) : (
        <View style={[styles.thumb, styles.thumbPlaceholder, { backgroundColor: c.surfaceMuted }]}>
          <Text style={[styles.thumbLetter, { color: c.textSecondary }]}>{title.slice(0, 1) || '?'}</Text>
        </View>
      )}
      <View style={styles.textCol}>
        <Text style={[styles.title, { color: c.text }]} numberOfLines={2}>
          {title}
        </Text>
        {hasNav && !hasAcq ? (
          <Text style={[styles.hint, { color: c.textSecondary }]}>Folder</Text>
        ) : hasAcq ? (
          <Text style={[styles.hint, { color: c.textSecondary }]}>
            {acquisitionFormat === 'pdf' ? 'PDF' : 'EPUB'}
          </Text>
        ) : null}
      </View>
    </Pressable>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 10,
    borderWidth: StyleSheet.hairlineWidth,
    marginBottom: 8,
  },
  thumb: { width: 48, height: 64, borderRadius: 4 },
  thumbPlaceholder: { alignItems: 'center', justifyContent: 'center' },
  thumbLetter: { fontSize: 20, fontWeight: '700' },
  textCol: { flex: 1, minWidth: 0 },
  title: { fontSize: 16, fontWeight: '600' },
  hint: { fontSize: 12, marginTop: 4 },
});
