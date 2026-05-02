import { Stack } from 'expo-router';
import { useMemo } from 'react';
import { BastionHeaderTitle } from '../../src/components/auth/BastionHeaderTitle';
import { useAppearancePreference } from '../../src/context/AppearancePreferenceContext';
import { getColors } from '../../src/theme/colors';

export default function AuthLayout() {
  const { resolvedScheme } = useAppearancePreference();
  const colors = getColors(resolvedScheme);

  const screenOptions = useMemo(
    () => ({
      headerShown: true,
      headerTitle: () => <BastionHeaderTitle />,
      headerTitleAlign: 'center' as const,
      headerStyle: { backgroundColor: colors.surface },
      headerShadowVisible: false,
      headerTintColor: colors.text,
      contentStyle: { backgroundColor: colors.background },
    }),
    [colors.surface, colors.text, colors.background]
  );

  return (
    <Stack screenOptions={screenOptions}>
      <Stack.Screen name="login" options={{ headerBackVisible: false }} />
      <Stack.Screen name="server" />
    </Stack>
  );
}
