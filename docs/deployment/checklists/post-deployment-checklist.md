# Post-Deployment Checklist

**Date**: ________________  
**Deployment Version**: ________________  
**Post-Deployment Start Time**: ________________  
**Reviewer**: ________________

## Phase 1: Immediate Verification (0-2 hours)

### 1. System Health
- [ ] All services healthy
  ```bash
  # Check all service statuses
  kubectl get pods -n production
  kubectl get deployments -n production
  ```

- [ ] No critical alerts firing
  ```bash
  # Check alert status
  curl http://alertmanager:9093/api/v1/alerts | jq '.data[] | select(.labels.severity=="critical")'
  ```

- [ ] Error rates within normal range
  ```bash
  # Check error rates
  ./scripts/check_error_rates.sh --window 2h
  ```

### 2. Performance Metrics
- [ ] Response times acceptable
  ```bash
  # Check latency metrics
  curl -s http://prometheus:9090/api/v1/query?query=http_request_duration_seconds{quantile="0.95"} | jq '.data.result[].value[1]'
  ```

- [ ] Throughput normal
  ```bash
  # Check request rate
  curl -s http://prometheus:9090/api/v1/query?query=rate(http_requests_total[5m]) | jq '.data.result[].value[1]'
  ```

- [ ] Resource utilization stable
  ```bash
  # Check resource usage
  kubectl top nodes
  kubectl top pods -n production
  ```

### 3. User Experience
- [ ] User complaints monitored
  - [ ] Support tickets checked
  - [ ] Social media monitored
  - [ ] Feedback channels reviewed

- [ ] Key user journeys tested
  ```bash
  # Run user journey tests
  pytest tests/e2e/user_journeys/ -v
  ```

## Phase 2: Extended Monitoring (2-24 hours)

### 1. Stability Monitoring
- [ ] No memory leaks detected
  ```python
  # Monitor memory usage over time
  python scripts/monitor_memory_usage.py --duration 24h --alert-threshold 90
  ```

- [ ] No gradual performance degradation
  ```bash
  # Track performance trends
  ./scripts/performance_trending.sh --hours 24
  ```

- [ ] Database performance stable
  ```sql
  -- Monitor slow queries
  SELECT query, mean_exec_time, calls
  FROM pg_stat_statements
  WHERE mean_exec_time > 1000
  ORDER BY mean_exec_time DESC;
  ```

### 2. Capacity Planning
- [ ] Auto-scaling working correctly
  ```bash
  # Verify auto-scaling events
  kubectl describe hpa musicgen-api-hpa -n production
  ```

- [ ] Resource usage within projections
  ```bash
  # Generate capacity report
  ./scripts/capacity_report.sh > capacity_report_$(date +%Y%m%d).txt
  ```

### 3. Error Analysis
- [ ] New error patterns identified
  ```bash
  # Analyze error logs
  ./scripts/analyze_errors.sh --since-deployment --group-by type
  ```

- [ ] Error remediation planned
  ```yaml
  # Document errors and fixes
  errors:
    - type: "ValidationError"
      count: 145
      cause: "New field validation"
      fix: "Update client libraries"
      priority: "medium"
  ```

## Phase 3: Business Verification (24-72 hours)

### 1. Business Metrics
- [ ] User engagement metrics normal
  ```sql
  -- Check user activity
  SELECT 
    DATE(created_at) as date,
    COUNT(DISTINCT user_id) as daily_active_users,
    COUNT(*) as total_requests
  FROM user_activity
  WHERE created_at >= NOW() - INTERVAL '3 days'
  GROUP BY DATE(created_at)
  ORDER BY date DESC;
  ```

- [ ] Conversion rates maintained
  ```python
  # Check conversion funnel
  python scripts/analyze_conversion.py --days 3 --compare-to-previous
  ```

- [ ] Revenue impact assessed
  ```bash
  # Generate revenue report
  ./scripts/revenue_impact.sh --deployment-date $(date +%Y-%m-%d)
  ```

### 2. Feature Adoption
- [ ] New features being used
  ```sql
  -- Track feature usage
  SELECT 
    feature_name,
    COUNT(DISTINCT user_id) as unique_users,
    COUNT(*) as total_uses
  FROM feature_usage
  WHERE created_at >= '${DEPLOYMENT_DATE}'
  GROUP BY feature_name
  ORDER BY total_uses DESC;
  ```

- [ ] A/B test results analyzed
  ```python
  # Analyze A/B tests
  python scripts/ab_test_analysis.py --deployment-version v1.0.0
  ```

## Phase 4: Documentation & Learning

### 1. Documentation Updates
- [ ] Deployment notes finalized
  ```markdown
  # Deployment Summary - v1.0.0
  
  ## What Went Well
  - 
  
  ## What Could Be Improved
  - 
  
  ## Action Items
  - 
  ```

- [ ] Runbooks updated
  - [ ] New error patterns added
  - [ ] Response procedures updated
  - [ ] Contact information verified

- [ ] Architecture diagrams updated
  ```bash
  # Regenerate architecture diagrams
  python scripts/generate_architecture_diagrams.py
  ```

### 2. Knowledge Sharing
- [ ] Post-mortem scheduled (if issues occurred)
  ```bash
  # Create post-mortem template
  ./scripts/create_postmortem.sh --deployment v1.0.0
  ```

- [ ] Lessons learned documented
  ```yaml
  lessons_learned:
    - category: "Process"
      lesson: "Pre-deployment testing caught critical bug"
      action: "Expand test coverage for edge cases"
    
    - category: "Technical"
      lesson: "Cache warming reduced post-deployment latency"
      action: "Automate cache warming in deployment script"
  ```

- [ ] Team knowledge base updated
  ```bash
  # Update team wiki
  ./scripts/update_wiki.sh --page deployments/v1.0.0
  ```

### 3. Process Improvements
- [ ] Deployment process reviewed
  - [ ] Time-consuming steps identified
  - [ ] Automation opportunities noted
  - [ ] Risk areas highlighted

- [ ] Tool improvements suggested
  ```markdown
  ## Tool Improvement Suggestions
  
  1. Automated rollback detection
     - Current: Manual monitoring required
     - Proposed: Automatic triggers based on SLOs
  
  2. Deployment preview environment
     - Current: Staging only
     - Proposed: Dynamic preview per PR
  ```

## Phase 5: Cleanup & Preparation

### 1. Resource Cleanup
- [ ] Temporary resources removed
  ```bash
  # Clean up deployment artifacts
  kubectl delete configmap maintenance-mode --ignore-not-found
  rm -f /tmp/deployment_*.log
  ```

- [ ] Old versions archived
  ```bash
  # Archive old images
  ./scripts/archive_old_versions.sh --keep-last 3
  ```

- [ ] Log rotation verified
  ```bash
  # Check log sizes
  du -sh /var/log/musicgen/*
  logrotate -f /etc/logrotate.d/musicgen
  ```

### 2. Security Review
- [ ] Access logs reviewed
  ```bash
  # Check for suspicious access
  ./scripts/analyze_access_logs.sh --since-deployment
  ```

- [ ] Temporary credentials revoked
  ```bash
  # Revoke deployment credentials
  ./scripts/revoke_temp_credentials.sh
  ```

- [ ] Security scan completed
  ```bash
  # Run post-deployment security scan
  ./scripts/security_scan.sh --full
  ```

### 3. Next Deployment Preparation
- [ ] Next version requirements gathered
- [ ] Technical debt items logged
  ```bash
  # Add to technical debt register
  ./scripts/log_tech_debt.sh --from-deployment v1.0.0
  ```

- [ ] Improvement backlog updated
  ```yaml
  improvements:
    - priority: high
      description: "Implement zero-downtime database migrations"
      effort: "1 sprint"
    
    - priority: medium
      description: "Add deployment progress API"
      effort: "3 days"
  ```

## Final Review

### Success Metrics
- [ ] Deployment SLOs met
  - Deployment time: _____ (target: <2 hours)
  - Downtime: _____ (target: 0)
  - Error spike: _____ (target: <1%)
  - Rollback required: Yes/No

- [ ] Customer impact assessment
  - Support tickets related to deployment: _____
  - Customer complaints: _____
  - Positive feedback: _____

### Approval Sign-offs
- [ ] Engineering Lead
  - Name: ________________
  - Date: ________________
  - Notes: ________________

- [ ] Operations Lead
  - Name: ________________
  - Date: ________________
  - Notes: ________________

- [ ] Product Owner
  - Name: ________________
  - Date: ________________
  - Notes: ________________

### Action Items

| Priority | Action | Owner | Due Date | Status |
|----------|--------|-------|----------|--------|
| High | | | | |
| Medium | | | | |
| Low | | | | |

### Recommendations for Next Deployment
```
1. 
2. 
3. 
```

---

**Post-Deployment Review Completed By**: ________________  
**Date/Time**: ________________  
**Next Deployment Scheduled**: ________________