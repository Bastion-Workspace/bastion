# Network Security Monitor Agent — Roosevelt's Digital Cavalry

**BULLY!** A dedicated security monitoring and automated response system deployed as an independent container with its own LoadBalancer IP for Kubernetes environments.

## Overview

The Network Security Monitor operates as Roosevelt's "Digital Cavalry" - a specialized security container that monitors network traffic, detects intrusion attempts, and executes automated reconnaissance and response operations. Deployed with its own LoadBalancer IP, it serves as a dedicated security perimeter that can listen and respond independently while coordinating with the main agent system.

## Architecture Strategy

### Container Independence
- **Dedicated Security Container**: Isolated from main application stack
- **Independent LoadBalancer IP**: Separate network identity for security operations
- **Pre-installed Security Tools**: Nmap, fail2ban, iptables, threat intelligence tools
- **Background Agent Communication**: Secure API channel to main Plato system

### Kubernetes Deployment Model
```yaml
# Security Monitor Service with LoadBalancer
apiVersion: v1
kind: Service
metadata:
  name: security-monitor-lb
spec:
  type: LoadBalancer
  selector:
    app: security-monitor
  ports:
  - port: 22
    targetPort: 22
    name: ssh-monitor
  - port: 8080
    targetPort: 8080
    name: security-api
```

## Primary Use Cases

### Active Network Monitoring
- **SSH Connection Monitoring**: Real-time detection of connection attempts on port 22
- **Failed Authentication Tracking**: Analysis of authentication failures and patterns
- **Brute Force Detection**: Identification of coordinated attack campaigns
- **Geographic Threat Analysis**: IP geolocation and regional threat assessment

### Automated Reconnaissance
- **Immediate IP Scanning**: Nmap reconnaissance of attacking sources
- **Service Fingerprinting**: Identification of attacker's exposed services
- **Vulnerability Assessment**: Detection of potential attack vectors on threat sources
- **Threat Intelligence Correlation**: Cross-reference with known threat databases

### Graduated Response System
- **Passive Monitoring**: Silent observation and data collection
- **Active Defense**: Dynamic firewall rules and rate limiting
- **Counter-Reconnaissance**: Authorized scanning of threat sources
- **Alert Escalation**: Human operator notification for critical threats

## Structured Output Models

### SecurityThreatEvent
```json
{
  "threat_id": "uuid",
  "timestamp": "ISO8601",
  "source_ip": "192.168.1.100",
  "target_port": 22,
  "attack_type": "ssh_brute_force|port_scan|unknown",
  "severity": "low|medium|high|critical",
  "evidence": ["failed_auth_attempt", "multiple_connections"],
  "geolocation": {
    "country": "US",
    "city": "New York",
    "isp": "Example ISP"
  },
  "confidence": 0.85
}
```

### NetworkScanResult
```json
{
  "target_ip": "192.168.1.100",
  "scan_timestamp": "ISO8601",
  "open_ports": [22, 80, 443],
  "services": {
    "22": "OpenSSH 8.2",
    "80": "Apache 2.4.41",
    "443": "Apache 2.4.41 (SSL)"
  },
  "vulnerabilities": ["CVE-2021-1234", "weak_ssh_config"],
  "threat_score": 0.75,
  "response_time_ms": 2500
}
```

### AutomatedResponseAction
```json
{
  "response_id": "uuid",
  "threat_id": "uuid",
  "action_type": "block_ip|rate_limit|honeypot_deploy|alert_only",
  "action_details": {
    "firewall_rule": "iptables -A INPUT -s 192.168.1.100 -j DROP",
    "duration_seconds": 3600,
    "escalation_required": false
  },
  "effectiveness": "successful|failed|pending",
  "human_review_required": true,
  "legal_compliance_checked": true
}
```

## Container Architecture

### Security Container Specifications
```dockerfile
# Roosevelt's Security Monitor Container
FROM ubuntu:22.04

# Install security tools
RUN apt-get update && apt-get install -y \
    nmap \
    fail2ban \
    iptables \
    netstat-nat \
    tcpdump \
    wireshark-common \
    python3 \
    python3-pip \
    openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Install Python security libraries
COPY requirements-security.txt /tmp/
RUN pip3 install -r /tmp/requirements-security.txt

# Security monitoring application
COPY security_monitor/ /app/
WORKDIR /app

# Expose SSH monitoring port and API port
EXPOSE 22 8080

CMD ["python3", "security_monitor.py"]
```

### Required Python Libraries
```txt
# requirements-security.txt
python-nmap==0.7.1
scapy==2.5.0
geoip2==4.7.0
requests==2.31.0
asyncio-mqtt==0.16.1
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
psutil==5.9.6
python-iptables==1.0.1
```

## Tool Integration

### Network Security Tools Registry
```python
# New tools for Security Agent
async def _register_network_security_tools(self):
    """Register network security and monitoring tools"""
    
    self._tools["monitor_ssh_attempts"] = ToolDefinition(
        name="monitor_ssh_attempts",
        function=monitor_ssh_connection_attempts,
        description="Monitor real-time SSH connection attempts and failures",
        access_level=ToolAccessLevel.NETWORK_SECURITY,
        parameters={
            "duration_seconds": {"type": "integer", "default": 60},
            "alert_threshold": {"type": "integer", "default": 5}
        }
    )
    
    self._tools["scan_suspicious_ip"] = ToolDefinition(
        name="scan_suspicious_ip", 
        function=perform_nmap_reconnaissance,
        description="Perform network reconnaissance on suspicious IP addresses",
        access_level=ToolAccessLevel.NETWORK_SECURITY,
        parameters={
            "target_ip": {"type": "string", "required": True},
            "scan_type": {"type": "string", "default": "stealth", "enum": ["stealth", "aggressive", "service_detection"]},
            "port_range": {"type": "string", "default": "1-1000"}
        }
    )
    
    self._tools["deploy_firewall_rule"] = ToolDefinition(
        name="deploy_firewall_rule",
        function=deploy_iptables_rule,
        description="Deploy automated firewall rules to block threats",
        access_level=ToolAccessLevel.NETWORK_SECURITY,
        parameters={
            "source_ip": {"type": "string", "required": True},
            "action": {"type": "string", "required": True, "enum": ["block", "rate_limit", "log_only"]},
            "duration_seconds": {"type": "integer", "default": 3600}
        }
    )
    
    self._tools["check_threat_intelligence"] = ToolDefinition(
        name="check_threat_intelligence",
        function=query_threat_databases,
        description="Check IP addresses against threat intelligence feeds",
        access_level=ToolAccessLevel.WEB_ACCESS,
        parameters={
            "ip_address": {"type": "string", "required": True},
            "databases": {"type": "array", "default": ["virustotal", "abuseipdb", "greynoise"]}
        }
    )
```

## Workflow Implementation

### Security Monitoring Workflow
```python
class SecurityMonitorWorkflow:
    """Roosevelt's Digital Cavalry Workflow"""
    
    async def continuous_monitoring_loop(self):
        """Main monitoring loop - runs continuously"""
        while True:
            # 1. Monitor SSH attempts
            ssh_events = await self.monitor_ssh_attempts(duration_seconds=30)
            
            # 2. Analyze each threat
            for event in ssh_events:
                threat_analysis = await self.analyze_threat_event(event)
                
                # 3. Execute graduated response
                if threat_analysis.severity in ["high", "critical"]:
                    await self.execute_immediate_response(threat_analysis)
                elif threat_analysis.severity == "medium":
                    await self.execute_enhanced_monitoring(threat_analysis)
                
            await asyncio.sleep(10)  # Brief pause between monitoring cycles
    
    async def execute_immediate_response(self, threat_analysis):
        """Execute immediate response for high-severity threats"""
        
        # 1. Reconnaissance phase
        scan_result = await self.scan_suspicious_ip(
            target_ip=threat_analysis.source_ip,
            scan_type="aggressive"
        )
        
        # 2. Threat intelligence check
        intel_result = await self.check_threat_intelligence(
            ip_address=threat_analysis.source_ip
        )
        
        # 3. Automated defensive action
        if intel_result.threat_score > 0.7:
            await self.deploy_firewall_rule(
                source_ip=threat_analysis.source_ip,
                action="block",
                duration_seconds=86400  # 24 hours
            )
        
        # 4. Alert human operators
        await self.escalate_to_human_review(threat_analysis, scan_result, intel_result)
```

## Kubernetes Integration

### Security Monitor Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: security-monitor
spec:
  replicas: 1
  selector:
    matchLabels:
      app: security-monitor
  template:
    metadata:
      labels:
        app: security-monitor
    spec:
      containers:
      - name: security-monitor
        image: plato/security-monitor:latest
        ports:
        - containerPort: 22
          name: ssh-monitor
        - containerPort: 8080
          name: security-api
        securityContext:
          privileged: true  # Required for network monitoring
          capabilities:
            add:
            - NET_ADMIN
            - NET_RAW
        env:
        - name: PLATO_API_ENDPOINT
          value: "http://plato-backend:8000"
        - name: SECURITY_MODE
          value: "active_defense"
        volumeMounts:
        - name: host-logs
          mountPath: /host/logs
          readOnly: true
      volumes:
      - name: host-logs
        hostPath:
          path: /var/log
```

### LoadBalancer Configuration
```yaml
apiVersion: v1
kind: Service
metadata:
  name: security-monitor-lb
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
spec:
  type: LoadBalancer
  selector:
    app: security-monitor
  ports:
  - port: 22
    targetPort: 22
    protocol: TCP
    name: ssh-honeypot
  - port: 8080
    targetPort: 8080
    protocol: TCP
    name: security-api
```

## Background Agent Integration

### Communication with Main Plato System
```python
class SecurityAgentBackgroundService:
    """Background service for security agent communication"""
    
    async def report_security_events(self, events: List[SecurityThreatEvent]):
        """Report security events to main Plato system"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PLATO_API_ENDPOINT}/api/security/events",
                json=[event.dict() for event in events],
                headers={"Authorization": f"Bearer {SECURITY_API_TOKEN}"}
            )
            return response.json()
    
    async def request_response_authorization(self, threat_analysis):
        """Request authorization for active response measures"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PLATO_API_ENDPOINT}/api/security/authorize",
                json={
                    "threat_id": threat_analysis.threat_id,
                    "requested_action": "active_reconnaissance",
                    "severity": threat_analysis.severity,
                    "evidence": threat_analysis.evidence
                }
            )
            return response.json().get("authorized", False)
```

## Security and Compliance

### Legal Compliance Framework
- **Defensive Operations**: Monitoring your own systems is fully legal
- **Passive Intelligence**: Gathering public information about attackers is acceptable
- **Active Scanning**: Requires explicit authorization and compliance review
- **Counter-Attacks**: Prohibited without legal authorization and oversight

### Data Protection
- **Encrypted Storage**: All threat intelligence encrypted at rest
- **Audit Logging**: Complete audit trail of all security actions
- **Data Retention**: Configurable retention policies for compliance
- **Privacy Protection**: No collection of personal data from legitimate users

## Monitoring and Alerting

### Real-time Dashboards
- **Threat Activity Map**: Geographic visualization of attack sources
- **Attack Pattern Analysis**: Trending and correlation of attack methods
- **Response Effectiveness**: Success rates of automated countermeasures
- **System Health**: Container and service health monitoring

### Alert Escalation
- **Immediate Alerts**: Critical threats requiring human intervention
- **Daily Summaries**: Comprehensive threat landscape reports
- **Weekly Analysis**: Pattern recognition and strategic recommendations
- **Monthly Reviews**: Security posture assessment and improvements

## Future Extensions

### Advanced Threat Hunting
- **Machine Learning**: Behavioral analysis for zero-day threat detection
- **Honeypot Networks**: Distributed deception systems
- **Threat Intelligence Sharing**: Integration with security communities
- **Forensic Analysis**: Deep packet inspection and attack reconstruction

### Integration Capabilities
- **SIEM Integration**: Export to Security Information and Event Management systems
- **Incident Response**: Automated ticket creation and workflow integration
- **Compliance Reporting**: Automated generation of security compliance reports
- **Threat Modeling**: Continuous assessment of attack surface and vulnerabilities

## Implementation Phases

### Phase 1: Foundation (2-3 weeks)
- ✅ Security container with basic monitoring capabilities
- ✅ SSH connection attempt detection and logging
- ✅ Basic IP scanning and reconnaissance tools
- ✅ Simple firewall rule automation

### Phase 2: Intelligence Integration (2-3 weeks)
- ✅ Threat intelligence feed integration
- ✅ Geographic and ISP analysis of attack sources
- ✅ Pattern recognition for attack campaigns
- ✅ Background communication with main Plato system

### Phase 3: Automated Response (2-3 weeks)
- ✅ Graduated response system implementation
- ✅ Dynamic firewall rule deployment
- ✅ Rate limiting and connection throttling
- ✅ Human authorization workflow for active measures

### Phase 4: Advanced Operations (3-4 weeks)
- ✅ Kubernetes LoadBalancer integration
- ✅ Advanced reconnaissance capabilities
- ✅ Honeypot deployment and management
- ✅ Comprehensive threat reporting and analytics

### Phase 5: Enterprise Features (ongoing)
- ✅ SIEM integration and compliance reporting
- ✅ Machine learning threat detection
- ✅ Distributed monitoring across multiple endpoints
- ✅ Advanced persistent threat tracking

## Technical Considerations

### Container Security
- **Privileged Access**: Required for network monitoring and firewall management
- **Network Capabilities**: NET_ADMIN and NET_RAW for packet capture
- **Host Log Access**: Read-only access to system authentication logs
- **Secure Communication**: Encrypted channels with main Plato system

### Performance Optimization
- **Efficient Packet Processing**: Optimized network monitoring with minimal overhead
- **Asynchronous Operations**: Non-blocking reconnaissance and response actions
- **Resource Management**: CPU and memory limits to prevent resource exhaustion
- **Log Rotation**: Automated cleanup of security logs and scan results

### Scalability Design
- **Horizontal Scaling**: Multiple security monitor instances for high-traffic environments
- **Load Distribution**: Intelligent distribution of monitoring tasks
- **Data Aggregation**: Centralized collection and analysis of security events
- **Geographic Distribution**: Multi-region deployment for global threat monitoring

## Legal and Ethical Framework

### Defensive Operations (Fully Legal)
- **Own System Monitoring**: Complete legal authority to monitor your own infrastructure
- **Log Analysis**: Analysis of connection attempts and authentication failures
- **Passive Intelligence**: Collection of publicly available information about attackers
- **Defensive Measures**: Blocking and rate limiting of malicious traffic

### Active Reconnaissance (Requires Authorization)
- **External IP Scanning**: Legal gray area requiring careful consideration
- **Service Fingerprinting**: May violate terms of service of target networks
- **Vulnerability Assessment**: Potential legal liability if misused
- **Authorization Required**: Explicit approval for any active measures

### Prohibited Activities
- **Offensive Operations**: No counter-attacks or offensive hacking
- **Data Theft**: No attempt to access or steal data from attackers
- **Service Disruption**: No denial of service attacks against threat sources
- **Unauthorized Access**: No attempts to gain unauthorized access to external systems

## Roosevelt's Security Doctrine

### "Speak Softly and Carry a Big Stick"
- **Silent Monitoring**: Observe and analyze without revealing defensive capabilities
- **Graduated Response**: Escalate responses proportionally to threat severity
- **Overwhelming Defense**: When action is required, respond decisively and comprehensively
- **Strategic Patience**: Gather intelligence before taking action

### "Trust but Verify"
- **Automated Systems**: Deploy intelligent automation for rapid response
- **Human Oversight**: Require human authorization for significant actions
- **Audit Everything**: Complete logging and audit trails for all security operations
- **Continuous Improvement**: Learn from each incident to strengthen defenses

## Environment Configuration

### Required Environment Variables
```bash
# Security Monitor Configuration
SECURITY_MODE=active_defense|passive_monitor|honeypot
PLATO_API_ENDPOINT=http://plato-backend:8000
SECURITY_API_TOKEN=secure_token_here
THREAT_INTEL_APIS=virustotal,abuseipdb,greynoise
MAX_SCAN_RATE=10_per_minute
FIREWALL_AUTO_BLOCK=true
HUMAN_AUTHORIZATION_REQUIRED=true

# Network Configuration  
MONITOR_INTERFACES=eth0,eth1
SSH_MONITOR_PORT=22
API_LISTEN_PORT=8080
LOG_RETENTION_DAYS=30

# Geographic and Legal
LEGAL_JURISDICTION=US
AUTHORIZED_SCAN_TARGETS=internal_only|authorized_external
COMPLIANCE_MODE=strict|moderate|permissive
```

### Security API Endpoints
```python
# Security Monitor API for background agent communication
@app.post("/api/security/events")
async def receive_security_events(events: List[SecurityThreatEvent])

@app.get("/api/security/status")
async def get_security_status() -> SecuritySystemStatus

@app.post("/api/security/authorize")
async def request_action_authorization(request: AuthorizationRequest) -> AuthorizationResponse

@app.get("/api/security/threats/{threat_id}")
async def get_threat_details(threat_id: str) -> ThreatAnalysisReport

@app.post("/api/security/response")
async def execute_authorized_response(action: AuthorizedSecurityAction)
```

## Background Tasks & Celery Integration

### Continuous Monitoring Tasks
```python
# Celery background tasks for security operations
@celery.task
def continuous_ssh_monitoring():
    """Background task for continuous SSH monitoring"""
    
@celery.task  
def threat_intelligence_updates():
    """Update threat intelligence databases"""
    
@celery.task
def security_report_generation():
    """Generate daily/weekly security reports"""
    
@celery.task
def firewall_rule_cleanup():
    """Clean up expired firewall rules"""
```

### LangGraph Integration
- **Node**: `security_monitor_agent` (real-time threat analysis)
- **Background**: `security.monitor_network`, `security.analyze_threats`, `security.generate_reports`
- **HITL Integration**: Human authorization for active response measures

## Deployment Strategy

### Docker Compose Development
```yaml
# Addition to docker-compose.yml
services:
  security-monitor:
    build: 
      context: ./security-monitor
      dockerfile: Dockerfile
    container_name: plato-security-monitor
    privileged: true
    network_mode: host  # Required for network monitoring
    ports:
      - "22:22"   # SSH monitoring port
      - "8080:8080"  # Security API
    environment:
      - SECURITY_MODE=passive_monitor  # Safe default for development
      - PLATO_API_ENDPOINT=http://backend:8000
    volumes:
      - /var/log:/host/logs:ro  # Read-only access to system logs
      - ./security-monitor/config:/app/config
    depends_on:
      - backend
      - postgres
```

### Production Kubernetes Deployment
- **Dedicated LoadBalancer IP**: Separate network identity for security operations
- **Privileged Security Context**: Required for network monitoring capabilities
- **Host Network Access**: Direct access to network interfaces and system logs
- **Secure Communication**: Encrypted API channels with main Plato system

## Roosevelt's Battle-Tested Security Principles

### The Four Pillars of Digital Defense

1. **Vigilant Monitoring**: "Eternal vigilance is the price of digital liberty!"
2. **Swift Intelligence**: "Know your enemy better than they know themselves!"
3. **Measured Response**: "Respond with precision, not with passion!"
4. **Legal Compliance**: "Fight honorably within the bounds of law!"

### The Security Cavalry Creed
- **"We monitor the digital frontier with unwavering dedication"**
- **"We gather intelligence with surgical precision"**  
- **"We defend with overwhelming force when justified"**
- **"We operate within the law and with honor"**

**BULLY!** This security monitoring system represents the finest in digital cavalry tactics - a well-organized, legally compliant, and devastatingly effective defense system that would make any digital frontier safe for honest settlers while keeping the bad actors at bay!

**By George!** With its own LoadBalancer IP and dedicated container, this security monitor can operate as an independent fortress while maintaining secure communication with the main cavalry regiment! 🏰⚔️🛡️
