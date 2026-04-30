import { FlatList, Pressable, StyleSheet, Text, View } from 'react-native';
import type { OpdsCatalogEntryResponse } from '../../api/ebooks';
import { getColors, type AppColors } from '../../theme/colors';

type Props = {
  catalogs: OpdsCatalogEntryResponse[];
  selectedId: string;
  onSelect: (id: string) => void;
  scheme: 'light' | 'dark' | null | undefined;
};

export function OpdsCatalogPicker({ catalogs, selectedId, onSelect, scheme }: Props) {
  const c: AppColors = getColors(scheme === 'dark' ? 'dark' : 'light');
  if (catalogs.length === 0) {
    return (
      <Text style={[styles.empty, { color: c.textSecondary }]}>
        Add an OPDS catalog in eBook settings to browse.
      </Text>
    );
  }
  return (
    <FlatList
      horizontal
      nestedScrollEnabled
      data={catalogs}
      keyExtractor={(cat) => cat.id}
      showsHorizontalScrollIndicator={false}
      style={styles.list}
      contentContainerStyle={styles.scroll}
      keyboardShouldPersistTaps="handled"
      renderItem={({ item: cat }) => {
        const selected = cat.id === selectedId;
        return (
          <Pressable
            onPress={() => onSelect(cat.id)}
            style={[
              styles.chip,
              { backgroundColor: selected ? c.chipBgActive : c.chipBg, borderColor: c.border },
            ]}
            accessibilityRole="button"
            accessibilityState={{ selected }}
          >
            <Text style={[styles.chipText, { color: selected ? c.chipTextActive : c.chipText }]} numberOfLines={1}>
              {cat.title || cat.root_url}
            </Text>
          </Pressable>
        );
      }}
      ItemSeparatorComponent={() => <View style={styles.sep} />}
    />
  );
}

const styles = StyleSheet.create({
  list: { flexGrow: 0, maxHeight: 52 },
  scroll: { paddingVertical: 4, paddingRight: 8, alignItems: 'center' },
  sep: { width: 8 },
  chip: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: StyleSheet.hairlineWidth,
    maxWidth: 200,
  },
  chipText: { fontSize: 14, fontWeight: '600' },
  empty: { fontSize: 14, paddingVertical: 8 },
});
