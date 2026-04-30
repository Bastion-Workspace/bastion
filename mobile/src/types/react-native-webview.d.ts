/**
 * Declarations so `tsc` passes before/without a full native install in restricted environments.
 * Runtime uses `react-native-webview` from package.json.
 */
declare module 'react-native-webview' {
  import type { Component, Ref } from 'react';
  import type { StyleProp, ViewStyle } from 'react-native';

  export interface WebViewProps {
    ref?: Ref<WebView>;
    style?: StyleProp<ViewStyle>;
    source: { uri: string } | { html: string; baseUrl?: string };
    originWhitelist?: string[];
    javaScriptEnabled?: boolean;
    domStorageEnabled?: boolean;
    setSupportMultipleWindows?: boolean;
    allowFileAccess?: boolean;
    allowFileAccessFromFileURLs?: boolean;
    allowUniversalAccessFromFileURLs?: boolean;
    mixedContentMode?: 'never' | 'always' | 'compatibility';
    onMessage?: (e: { nativeEvent: { data: string } }) => void;
  }

  export default class WebView extends Component<WebViewProps> {
    injectJavaScript(script: string): void;
  }
}
