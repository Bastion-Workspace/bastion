import { Image } from 'expo-image';
import { StyleSheet, View } from 'react-native';
import { useAppearancePreference } from '../../context/AppearancePreferenceContext';

/** Same marks as web PWA splash; bundled via Metro watchFolders → ../frontend/public/images */
const logoLight = require('../../../../frontend/public/images/bastion.png');
const logoDark = require('../../../../frontend/public/images/bastion-dark.png');

export function BastionHeaderTitle() {
  const { resolvedScheme } = useAppearancePreference();
  const isDark = resolvedScheme === 'dark';
  const src = isDark ? logoDark : logoLight;

  return (
    <View style={styles.wrap} accessibilityRole="header" accessibilityLabel="Bastion">
      <Image
        source={src}
        style={styles.img}
        contentFit="contain"
        cachePolicy="memory"
        accessibilityIgnoresInvertColors
      />
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    height: 30,
    minWidth: 100,
    maxWidth: 200,
    alignSelf: 'center',
    justifyContent: 'center',
    alignItems: 'center',
  },
  img: {
    height: 28,
    width: 168,
  },
});
