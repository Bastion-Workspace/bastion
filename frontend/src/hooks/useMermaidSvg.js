import { useEffect, useState } from 'react';

/**
 * Renders Mermaid source to SVG via dynamic import. Matches chat artifact behavior
 * (securityLevel strict, theme from darkMode).
 *
 * @param {string} source
 * @param {{ darkMode: boolean, enabled?: boolean }} options
 */
export function useMermaidSvg(source, { darkMode, enabled = true }) {
  const [svg, setSvg] = useState('');
  const [error, setError] = useState('');

  useEffect(() => {
    if (!enabled) {
      setSvg('');
      setError('');
      return undefined;
    }

    let cancelled = false;
    setError('');
    setSvg('');
    const trimmed = String(source ?? '').trim();
    if (!trimmed) {
      setError('Empty diagram source');
      return undefined;
    }

    const id = `mermaid-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
    import('mermaid')
      .then((mod) => {
        if (cancelled) return;
        const mermaid = mod.default;
        mermaid.initialize({
          startOnLoad: false,
          securityLevel: 'strict',
          theme: darkMode ? 'dark' : 'neutral',
        });
        return mermaid.render(id, trimmed);
      })
      .then((result) => {
        if (cancelled || !result) return;
        setSvg(result.svg || '');
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err?.message || 'Failed to render diagram');
        }
      });

    return () => {
      cancelled = true;
    };
  }, [source, darkMode, enabled]);

  const trimmed = String(source ?? '').trim();
  const loading = enabled && !error && !svg && trimmed.length > 0;

  return { svg, error, loading };
}
