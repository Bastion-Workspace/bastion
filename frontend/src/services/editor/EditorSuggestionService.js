import ApiServiceBase from '../base/ApiServiceBase';

class EditorSuggestionService extends ApiServiceBase {
  async suggest({ prefix, suffix, filename, language = 'markdown', cursorOffset = -1, frontmatter = null, maxChars = 300, signal }) {
    const body = {
      prefix: String(prefix || ''),
      suffix: String(suffix || ''),
      filename,
      language,
      cursor_offset: cursorOffset,
      frontmatter,
      max_chars: maxChars,
      temperature: 0.25
    };
    try {
      return await this.post('/api/editor/suggest', body, { signal });
    } catch (e) {
      return { suggestion: '', confidence: 0, model_used: null };
    }
  }
}

export const editorSuggestionService = new EditorSuggestionService();


