import ApiServiceBase from '../base/ApiServiceBase';

class NotesService extends ApiServiceBase {
  // Notes methods
  getNotes = async (skip = 0, limit = 100, category = null, tag = null, search = null) => {
    let url = `/api/notes?skip=${skip}&limit=${limit}`;
    if (category) url += `&category=${encodeURIComponent(category)}`;
    if (tag) url += `&tag=${encodeURIComponent(tag)}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;
    return this.get(url);
  }

  getNote = async (noteId) => {
    return this.get(`/api/notes/${noteId}`);
  }

  createNote = async (noteData) => {
    return this.post('/api/notes', noteData);
  }

  updateNote = async (noteId, noteData) => {
    return this.put(`/api/notes/${noteId}`, noteData);
  }

  deleteNote = async (noteId) => {
    return this.delete(`/api/notes/${noteId}`);
  }

  searchNotes = async (query, limit = 100) => {
    return this.post('/api/notes/search', {
      query: query,
      limit: limit
    });
  }

  getNoteTags = async () => {
    return this.get('/api/notes/categories');
  }

  exportNotes = async () => {
    const response = await fetch(`${this.baseURL}/api/notes/export`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('auth_token') || ''}`
      }
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    return response.blob();
  }


}

export default new NotesService();
