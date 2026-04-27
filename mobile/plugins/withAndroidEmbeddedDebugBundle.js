/**
 * Embed the JS bundle in debug APKs (assembleDebug) so installs work without Metro.
 * Default RN/Expo treats "debug" as debuggableVariants and skips bundling; CI has no packager.
 * @see https://reactnative.dev/docs/react-native-gradle-plugin#debuggablevariants
 */
const { withAppBuildGradle } = require('expo/config-plugins');

module.exports = function withAndroidEmbeddedDebugBundle(config) {
  return withAppBuildGradle(config, (mod) => {
    if (mod.modResults.language !== 'groovy') {
      return mod;
    }
    let contents = mod.modResults.contents;
    // Expo template comments mention debuggableVariants; do not use includes() or we skip injection.
    if (/^\s*debuggableVariants\s*=/m.test(contents)) {
      return mod;
    }
    mod.modResults.contents = contents.replace(
      /react\s*\{/,
      `react {
    // Embed JS for debug builds (standalone APK from CI; no Metro on device).
    debuggableVariants = []`
    );
    return mod;
  });
};
