import ApiServiceBase from '../base/ApiServiceBase';

class AudioService extends ApiServiceBase {
  transcribeAudio = async (blob) => {
    const token = localStorage.getItem('auth_token');

    const form = new FormData();
    // Provide a filename for better server/provider handling
    const file = new File([blob], 'recording.webm', { type: blob.type || 'audio/webm' });
    form.append('file', file);

    const response = await fetch(`${this.baseURL}/api/audio/transcribe`, {
      method: 'POST',
      headers: {
        ...(token ? { 'Authorization': `Bearer ${token}` } : {})
        // Do not set Content-Type for FormData; browser sets boundary
      },
      body: form
    });

    if (!response.ok) {
      let detail = 'Transcription failed';
      try {
        const data = await response.json();
        detail = data.detail || JSON.stringify(data);
      } catch {}
      throw new Error(detail);
    }

    const data = await response.json();
    if (!data.success) {
      throw new Error(data.detail || 'Transcription service error');
    }
    return data.text || '';
  }
}

export default new AudioService();


