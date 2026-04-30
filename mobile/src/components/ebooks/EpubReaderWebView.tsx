import { forwardRef, useCallback, useImperativeHandle, useMemo, useRef } from 'react';
import { StyleSheet } from 'react-native';
import WebView from 'react-native-webview';

export type ReaderTheme = 'light' | 'sepia' | 'dark';

export type EpubReaderCommand =
  | { type: 'PREV' }
  | { type: 'NEXT' }
  | { type: 'GOTO_CFI'; cfi: string }
  | { type: 'SET_THEME'; theme: ReaderTheme; fontSize?: number }
  | { type: 'SET_FONT_SIZE'; fontSize: number; theme?: ReaderTheme }
  | { type: 'SEEK_PERCENT'; pct: number };

export type WebToNativeMessage =
  | { type: 'READY' }
  | { type: 'RELOCATED'; cfi: string; percentage: number }
  | { type: 'ERROR'; message: string };

type Props = {
  sourceHtml: string;
  onMessage: (msg: WebToNativeMessage) => void;
};

export type EpubReaderWebViewHandle = {
  sendCommand: (cmd: EpubReaderCommand) => void;
};

export const EpubReaderWebView = forwardRef<EpubReaderWebViewHandle, Props>(function EpubReaderWebViewFn(
  props: Props,
  ref
) {
  const { sourceHtml, onMessage } = props;
  const webRef = useRef<WebView | null>(null);

  const sendCommand = useCallback((cmd: EpubReaderCommand) => {
    const payload = JSON.stringify(cmd);
    const js = `(function(){try{window.receiveCmd(${payload});}catch(e){}true;})();`;
    webRef.current?.injectJavaScript(js);
  }, []);

  useImperativeHandle(ref, () => ({ sendCommand }), [sendCommand]);

  const onWebMessage = useCallback(
    (e: { nativeEvent: { data: string } }) => {
      try {
        const raw = e.nativeEvent.data;
        const parsed = JSON.parse(raw) as WebToNativeMessage;
        if (parsed && typeof parsed.type === 'string') {
          onMessage(parsed);
        }
      } catch {
        // ignore
      }
    },
    [onMessage]
  );

  const originWhitelist = useMemo(() => ['*'], []);

  return (
    <WebView
      ref={webRef}
      style={styles.web}
      source={{ html: sourceHtml, baseUrl: '' }}
      originWhitelist={originWhitelist}
      onMessage={onWebMessage}
      javaScriptEnabled
      domStorageEnabled
      mixedContentMode="always"
      setSupportMultipleWindows={false}
    />
  );
});

const styles = StyleSheet.create({
  web: { flex: 1, backgroundColor: 'transparent' },
});
