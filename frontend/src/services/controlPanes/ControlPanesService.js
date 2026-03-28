import ApiServiceBase from '../base/ApiServiceBase';

const CONTROL_PANES_PREFIX = '/api/control-panes';

class ControlPanesService extends ApiServiceBase {
  listPanes = () => this.get(CONTROL_PANES_PREFIX);

  getPane = (id) => this.get(`${CONTROL_PANES_PREFIX}/${id}`);

  createPane = (data) => this.post(CONTROL_PANES_PREFIX, data);

  updatePane = (id, data) => this.put(`${CONTROL_PANES_PREFIX}/${id}`, data);

  deletePane = (id) => this.delete(`${CONTROL_PANES_PREFIX}/${id}`);

  toggleVisibility = (id, isVisible) =>
    this.patch(`${CONTROL_PANES_PREFIX}/${id}/visibility`, { is_visible: isVisible });

  executeAction = (paneId, endpointId, params = {}) =>
    this.post(`${CONTROL_PANES_PREFIX}/${paneId}/execute`, {
      endpoint_id: endpointId,
      params: params || {},
    });

  testEndpoint = (body) =>
    this.post(`${CONTROL_PANES_PREFIX}/test-endpoint`, body);
}

export default new ControlPanesService();
