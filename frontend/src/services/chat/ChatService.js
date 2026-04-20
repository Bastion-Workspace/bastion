import ApiServiceBase from '../base/ApiServiceBase';

class ChatService extends ApiServiceBase {
  // Cancel unified job
  cancelUnifiedJob = async (jobId) => {
    return this.post(`/api/v2/chat/unified/job/${jobId}/cancel`);
  }
}

export default new ChatService();
