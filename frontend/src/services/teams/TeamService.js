import ApiServiceBase from '../base/ApiServiceBase';

class TeamService extends ApiServiceBase {
  // Team Management
  async createTeam(teamData) {
    return this.post('/api/teams', teamData);
  }

  async getTeams() {
    return this.get('/api/teams');
  }

  async getTeam(teamId) {
    return this.get(`/api/teams/${teamId}`);
  }

  async updateTeam(teamId, updates) {
    return this.put(`/api/teams/${teamId}`, updates);
  }

  async deleteTeam(teamId) {
    return this.delete(`/api/teams/${teamId}`);
  }

  // Member Management
  async getTeamMembers(teamId) {
    return this.get(`/api/teams/${teamId}/members`);
  }

  async addMember(teamId, userId, role = 'member') {
    return this.post(`/api/teams/${teamId}/members`, { user_id: userId, role });
  }

  async removeMember(teamId, userId) {
    return this.delete(`/api/teams/${teamId}/members/${userId}`);
  }

  async updateMemberRole(teamId, userId, role) {
    return this.put(`/api/teams/${teamId}/members/${userId}/role`, { role });
  }

  // Invitations
  async createInvitation(teamId, invitedUserId) {
    return this.post(`/api/teams/${teamId}/invitations?invited_user_id=${invitedUserId}`);
  }

  async getTeamInvitations(teamId) {
    return this.get(`/api/teams/${teamId}/invitations`);
  }

  async cancelInvitation(teamId, invitationId) {
    return this.delete(`/api/teams/${teamId}/invitations/${invitationId}`);
  }

  async getPendingInvitations() {
    return this.get('/api/teams/invitations/pending');
  }

  async acceptInvitation(invitationId) {
    return this.put(`/api/teams/invitations/${invitationId}/accept`);
  }

  async rejectInvitation(invitationId) {
    return this.put(`/api/teams/invitations/${invitationId}/reject`);
  }

  // Posts
  async getTeamPosts(teamId, limit = 20, beforePostId = null) {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (beforePostId) {
      params.append('before_post_id', beforePostId);
    }
    return this.get(`/api/teams/${teamId}/posts?${params.toString()}`);
  }

  async createPost(teamId, content, postType = 'text', attachments = []) {
    return this.post(`/api/teams/${teamId}/posts`, {
      content,
      post_type: postType,
      attachments
    });
  }

  async deletePost(teamId, postId) {
    return this.delete(`/api/teams/${teamId}/posts/${postId}`);
  }

  // Reactions
  async addReaction(teamId, postId, reactionType) {
    return this.post(`/api/teams/${teamId}/posts/${postId}/reactions`, {
      reaction_type: reactionType
    });
  }

  async removeReaction(teamId, postId, reactionType) {
    return this.delete(`/api/teams/${teamId}/posts/${postId}/reactions/${reactionType}`);
  }

  // Comments
  async getPostComments(teamId, postId, limit = 50) {
    return this.get(`/api/teams/${teamId}/posts/${postId}/comments?limit=${limit}`);
  }

  async createComment(teamId, postId, content) {
    return this.post(`/api/teams/${teamId}/posts/${postId}/comments`, { content });
  }

  async deleteComment(teamId, postId, commentId) {
    return this.delete(`/api/teams/${teamId}/posts/${postId}/comments/${commentId}`);
  }

  // Unread Tracking
  async getUnreadPostCounts() {
    return this.get('/api/teams/unread-counts');
  }

  async markTeamPostsAsRead(teamId) {
    return this.post(`/api/teams/${teamId}/mark-read`);
  }

  async muteTeam(teamId, muted = true) {
    return this.put(`/api/teams/${teamId}/mute?muted=${muted}`);
  }
}

export default new TeamService();

