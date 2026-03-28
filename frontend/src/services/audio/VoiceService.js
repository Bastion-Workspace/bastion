import ApiServiceBase from '../base/ApiServiceBase';

class VoiceService extends ApiServiceBase {
  availabilityCache = null;

  availabilityCacheAt = 0;

  availabilityTtlMs = 30000;

  synthesize = async (text, options = {}) => {
    const { signal, ...voiceOpts } = options;
    const payload = {
      text: String(text || ''),
      voice_id: voiceOpts.voiceId || '',
      provider: voiceOpts.provider || '',
      output_format: voiceOpts.outputFormat || 'mp3',
    };

    if (!payload.text.trim()) {
      throw new Error('Text is required for synthesis');
    }

    const token = localStorage.getItem('auth_token');
    const response = await fetch(`${this.baseURL}/api/voice/synthesize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(payload),
      ...(signal ? { signal } : {}),
    });

    if (!response.ok) {
      let detail = 'Voice synthesis failed';
      try {
        const data = await response.json();
        detail = data.detail || JSON.stringify(data);
      } catch {}
      throw new Error(detail);
    }

    const audioBlob = await response.blob();
    const audioFormat = response.headers.get('X-Audio-Format') || payload.output_format || 'mp3';

    return { blob: audioBlob, format: audioFormat };
  };

  /**
   * POST streaming TTS. Returns the fetch Response (body ReadableStream) or throws.
   * Prefer outputFormat 'ogg' for MediaSource (Opus in OGG).
   */
  fetchSynthesizeStream = async (text, options = {}) => {
    const { signal, ...voiceOpts } = options;
    const payload = {
      text: String(text || ''),
      voice_id: voiceOpts.voiceId || '',
      provider: voiceOpts.provider || '',
      output_format: voiceOpts.outputFormat || 'ogg',
    };

    if (!payload.text.trim()) {
      throw new Error('Text is required for synthesis');
    }

    const token = localStorage.getItem('auth_token');
    const response = await fetch(`${this.baseURL}/api/voice/synthesize/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(payload),
      ...(signal ? { signal } : {}),
    });

    if (!response.ok) {
      let detail = 'Voice synthesis stream failed';
      try {
        const data = await response.json();
        detail = data.detail || JSON.stringify(data);
      } catch {
        try {
          detail = await response.text();
        } catch {
          /* ignore */
        }
      }
      throw new Error(detail);
    }

    return response;
  };

  listVoices = async (provider = '') => {
    const query = provider ? `?provider=${encodeURIComponent(provider)}` : '';
    const response = await this.get(`/api/voice/voices${query}`);
    return response?.voices || [];
  };

  /**
   * Synchronous read of last checkAvailability result (for hook initial state).
   * @returns {boolean|null} true/false if cache fresh, else null
   */
  getAvailabilityFromCache = () => {
    const now = Date.now();
    if (!this.availabilityCache || now - this.availabilityCacheAt >= this.availabilityTtlMs) {
      return null;
    }
    return !!this.availabilityCache.available;
  };

  /**
   * Start server-side audio export (Celery). Returns { task_id, status }.
   */
  startAudioExport = async (payload, options = {}) => {
    const { signal } = options;
    return this.post(
      '/api/audio-export',
      {
        document_id: payload.documentId,
        provider: payload.provider || '',
        voice_id: payload.voiceId || '',
      },
      signal ? { signal } : {}
    );
  };

  getAudioExportStatus = async (taskId, options = {}) => {
    const { signal } = options;
    const path = `/api/audio-export/${encodeURIComponent(taskId)}/status`;
    return signal ? this.get(path, { signal }) : this.get(path);
  };

  downloadAudioExportBlob = async (taskId, options = {}) => {
    const { signal } = options;
    const token = localStorage.getItem('auth_token');
    const response = await fetch(
      `${this.baseURL}/api/audio-export/${encodeURIComponent(taskId)}/download`,
      {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        ...(signal ? { signal } : {}),
      }
    );
    if (!response.ok) {
      let detail = 'Audio export download failed';
      try {
        const data = await response.json();
        detail = data.detail || JSON.stringify(data);
      } catch {
        try {
          detail = await response.text();
        } catch {
          /* ignore */
        }
      }
      throw new Error(detail);
    }
    return response.blob();
  };

  checkAvailability = async (forceRefresh = false) => {
    const now = Date.now();
    if (!forceRefresh && this.availabilityCache && now - this.availabilityCacheAt < this.availabilityTtlMs) {
      return this.availabilityCache;
    }

    try {
      const response = await this.get('/api/voice/status');
      const normalized = {
        available: !!response?.available,
        status: response?.status || 'unknown',
        providers: Array.isArray(response?.providers) ? response.providers : [],
        voiceCount: Number(response?.voice_count || 0),
      };
      this.availabilityCache = normalized;
      this.availabilityCacheAt = now;
      return normalized;
    } catch {
      const unavailable = {
        available: false,
        status: 'unavailable',
        providers: [],
        voiceCount: 0,
      };
      this.availabilityCache = unavailable;
      this.availabilityCacheAt = now;
      return unavailable;
    }
  };
}

export default new VoiceService();
