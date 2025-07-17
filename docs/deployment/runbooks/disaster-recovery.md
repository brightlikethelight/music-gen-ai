# Disaster Recovery Runbook

**Purpose**: This runbook provides step-by-step procedures for recovering the Music Gen AI system from various disaster scenarios.

**Last Updated**: January 2024  
**Owner**: Platform Team  
**Review Frequency**: Quarterly

## Table of Contents

1. [Disaster Scenarios](#disaster-scenarios)
2. [Recovery Time Objectives](#recovery-time-objectives)
3. [Complete System Failure](#complete-system-failure)
4. [Data Center Outage](#data-center-outage)
5. [Database Corruption](#database-corruption)
6. [Ransomware Attack](#ransomware-attack)
7. [Critical Data Loss](#critical-data-loss)
8. [Regional Disaster](#regional-disaster)

---

## Disaster Scenarios

| Scenario | Severity | RTO | RPO | Page |
|----------|----------|-----|-----|------|
| Complete System Failure | Critical | 2 hours | 1 hour | [Link](#complete-system-failure) |
| Data Center Outage | Critical | 4 hours | 1 hour | [Link](#data-center-outage) |
| Database Corruption | High | 2 hours | 1 hour | [Link](#database-corruption) |
| Ransomware Attack | Critical | 8 hours | 24 hours | [Link](#ransomware-attack) |
| Critical Data Loss | High | 4 hours | 1 hour | [Link](#critical-data-loss) |
| Regional Disaster | Critical | 8 hours | 4 hours | [Link](#regional-disaster) |

---

## Recovery Time Objectives

### Service Priority Tiers

| Tier | Services | RTO | RPO |
|------|----------|-----|-----|
| 1 | API Gateway, Auth Service | 1 hour | 15 min |
| 2 | Generation Service, Database | 2 hours | 1 hour |
| 3 | Model Serving, Cache Layer | 4 hours | 1 hour |
| 4 | Analytics, Monitoring | 8 hours | 24 hours |
| 5 | Development/Staging | 24 hours | 7 days |

---

## Complete System Failure

**Scenario**: All production systems are down due to cascading failure, misconfiguration, or cyberattack.

### Phase 1: Assessment (0-15 minutes)

```bash
#!/bin/bash
# Initial assessment script

echo "=== System Failure Assessment ==="
echo "Time: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# 1. Check all regions
for region in us-east-1 us-west-2 eu-west-1; do
    echo "Checking region: $region"
    aws ec2 describe-instance-status --region $region \
        --filters Name=instance-state-name,Values=running | \
        jq '.InstanceStatuses | length'
done

# 2. Check critical services
SERVICES=("api.musicgen.ai" "app.musicgen.ai" "postgres.musicgen.ai")
for service in "${SERVICES[@]}"; do
    if ping -c 1 -W 2 $service >/dev/null 2>&1; then
        echo "✅ $service is reachable"
    else
        echo "❌ $service is DOWN"
    fi
done

# 3. Check last known good backup
aws s3 ls s3://musicgen-backups/ --recursive | \
    grep "full-backup" | \
    sort -r | \
    head -1
```

### Phase 2: Communication (15-30 minutes)

1. **Activate Crisis Team**
   ```bash
   # Send emergency notification
   ./scripts/activate_crisis_team.sh "Complete system failure detected"
   ```

2. **Update Status Page**
   ```bash
   curl -X POST https://api.statuspage.io/v1/pages/${PAGE_ID}/incidents \
     -H "Authorization: OAuth ${STATUSPAGE_API_KEY}" \
     -d '{
       "incident": {
         "name": "Major Service Outage",
         "status": "investigating",
         "impact_override": "critical",
         "body": "We are experiencing a complete service outage. Our team is investigating."
       }
     }'
   ```

3. **Internal Communication**
   - Slack: Post in #incident-response
   - Email: Send to all-hands@musicgen.ai
   - War Room: Create video conference link

### Phase 3: Recovery Execution (30 minutes - 2 hours)

#### Option A: Restore from Backup (Recommended)

```bash
#!/bin/bash
# Full system restoration script

set -euo pipefail

# Configuration
BACKUP_DATE=$(date -d "1 hour ago" +%Y%m%d_%H)
S3_BUCKET="musicgen-backups"
RECOVERY_REGION="us-east-1"

echo "Starting full system recovery from backup: ${BACKUP_DATE}"

# 1. Provision new infrastructure
echo "Provisioning infrastructure in ${RECOVERY_REGION}..."
cd infrastructure/terraform
terraform workspace select disaster-recovery
terraform apply -auto-approve -var="region=${RECOVERY_REGION}"

# 2. Restore database
echo "Restoring database..."
BACKUP_FILE="s3://${S3_BUCKET}/${BACKUP_DATE}/database.dump"
aws s3 cp $BACKUP_FILE /tmp/database.dump

# Create new RDS instance from snapshot
DB_SNAPSHOT=$(aws rds describe-db-snapshots \
    --query "DBSnapshots[?SnapshotCreateTime>=\`${BACKUP_DATE}\`] | [0].DBSnapshotIdentifier" \
    --output text)

aws rds restore-db-instance-from-db-snapshot \
    --db-instance-identifier musicgen-recovery \
    --db-snapshot-identifier $DB_SNAPSHOT \
    --db-instance-class db.r6i.4xlarge \
    --multi-az

# Wait for database
echo "Waiting for database to be available..."
aws rds wait db-instance-available \
    --db-instance-identifier musicgen-recovery

# 3. Deploy applications
echo "Deploying applications..."
kubectl config use-context disaster-recovery

# Deploy core services
kubectl apply -f k8s/disaster-recovery/namespace.yaml
kubectl apply -f k8s/disaster-recovery/configs/
kubectl apply -f k8s/disaster-recovery/deployments/

# 4. Restore data
echo "Restoring application data..."
aws s3 sync s3://${S3_BUCKET}/${BACKUP_DATE}/models/ /mnt/models/
aws s3 sync s3://${S3_BUCKET}/${BACKUP_DATE}/audio/ /mnt/audio/

# 5. Update DNS
echo "Updating DNS to point to recovery environment..."
./scripts/update_dns.sh disaster-recovery

# 6. Verify services
echo "Verifying service health..."
./scripts/health_check_all.sh disaster-recovery

echo "Recovery completed at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
```

#### Option B: Failover to Secondary Region

```bash
#!/bin/bash
# Regional failover script

# 1. Activate secondary region
aws route53 change-resource-record-sets \
    --hosted-zone-id Z1234567890ABC \
    --change-batch '{
      "Changes": [{
        "Action": "UPSERT",
        "ResourceRecordSet": {
          "Name": "api.musicgen.ai",
          "Type": "A",
          "AliasTarget": {
            "HostedZoneId": "Z2FDTNDATAQYW2",
            "DNSName": "dr-alb.us-west-2.elb.amazonaws.com",
            "EvaluateTargetHealth": true
          }
        }
      }]
    }'

# 2. Scale up secondary region
kubectl config use-context us-west-2
kubectl scale deployment --all -n production --replicas=5

# 3. Verify traffic shifting
watch -n 5 'curl -s https://api.musicgen.ai/health | jq .region'
```

### Phase 4: Validation (2-3 hours)

```bash
#!/bin/bash
# Post-recovery validation

# 1. Service health checks
for endpoint in api app cdn metrics; do
    response=$(curl -s -w "\n%{http_code}" https://${endpoint}.musicgen.ai/health)
    status_code=$(echo "$response" | tail -n1)
    if [ "$status_code" = "200" ]; then
        echo "✅ ${endpoint}: Healthy"
    else
        echo "❌ ${endpoint}: Unhealthy (${status_code})"
    fi
done

# 2. Run smoke tests
pytest tests/disaster_recovery/smoke_tests.py -v

# 3. Check data integrity
python scripts/verify_data_integrity.py \
    --check-users \
    --check-generations \
    --check-models

# 4. Monitor error rates
./scripts/monitor_recovery.sh --duration 30m
```

---

## Data Center Outage

**Scenario**: Primary data center (AWS us-east-1) is completely unavailable.

### Immediate Actions (0-30 minutes)

```bash
#!/bin/bash
# Data center failover script

# 1. Confirm outage
aws ec2 describe-regions --region us-west-2
aws health describe-events --region us-east-1 || echo "Region unreachable"

# 2. Activate DR site
export DR_REGION="us-west-2"
export DR_CLUSTER="musicgen-dr"

# 3. Update global load balancer
aws globalaccelerator update-endpoint-group \
    --endpoint-group-arn $ENDPOINT_GROUP_ARN \
    --endpoint-configurations '[
      {
        "EndpointId": "dr-alb-1234567890.us-west-2.elb.amazonaws.com",
        "Weight": 100,
        "ClientIPPreservationEnabled": true
      }
    ]'

# 4. Promote DR database to primary
aws rds promote-read-replica \
    --db-instance-identifier musicgen-dr-replica \
    --backup-retention-period 7 \
    --preferred-backup-window "03:00-04:00"
```

### Database Failover (30-60 minutes)

```python
#!/usr/bin/env python3
"""
Database failover orchestration for regional disaster
"""
import boto3
import time
import psycopg2
from datetime import datetime

class RegionalDatabaseFailover:
    def __init__(self, primary_region, dr_region):
        self.primary_region = primary_region
        self.dr_region = dr_region
        self.rds_primary = boto3.client('rds', region_name=primary_region)
        self.rds_dr = boto3.client('rds', region_name=dr_region)
        
    def execute_failover(self):
        print(f"Starting regional database failover at {datetime.utcnow()}")
        
        # 1. Check if primary is truly down
        try:
            self.rds_primary.describe_db_instances(
                DBInstanceIdentifier='musicgen-primary'
            )
            print("WARNING: Primary appears to be up. Confirm failover?")
            # In real scenario, add confirmation logic
        except Exception as e:
            print(f"Primary region unreachable: {e}")
        
        # 2. Promote DR replica
        print("Promoting DR replica to primary...")
        try:
            response = self.rds_dr.promote_read_replica(
                DBInstanceIdentifier='musicgen-dr-replica',
                BackupRetentionPeriod=7
            )
            print(f"Promotion initiated: {response['DBInstance']['DBInstanceIdentifier']}")
        except Exception as e:
            print(f"Error promoting replica: {e}")
            return False
        
        # 3. Wait for promotion
        print("Waiting for promotion to complete...")
        waiter = self.rds_dr.get_waiter('db_instance_available')
        waiter.wait(
            DBInstanceIdentifier='musicgen-dr-replica',
            WaiterConfig={'Delay': 30, 'MaxAttempts': 40}
        )
        
        # 4. Update application configuration
        self.update_app_config()
        
        # 5. Verify new primary
        if self.verify_new_primary():
            print("✅ Failover completed successfully")
            return True
        else:
            print("❌ Failover verification failed")
            return False
    
    def update_app_config(self):
        """Update application configuration for new primary"""
        ssm = boto3.client('ssm', region_name=self.dr_region)
        
        new_endpoint = self.get_new_endpoint()
        
        ssm.put_parameter(
            Name='/musicgen/database/primary_endpoint',
            Value=new_endpoint,
            Type='String',
            Overwrite=True
        )
        
        # Trigger application restart
        ecs = boto3.client('ecs', region_name=self.dr_region)
        ecs.update_service(
            cluster='musicgen-dr',
            service='musicgen-api',
            forceNewDeployment=True
        )
    
    def get_new_endpoint(self):
        """Get the endpoint of the newly promoted primary"""
        response = self.rds_dr.describe_db_instances(
            DBInstanceIdentifier='musicgen-dr-replica'
        )
        return response['DBInstances'][0]['Endpoint']['Address']
    
    def verify_new_primary(self):
        """Verify the new primary is accepting writes"""
        endpoint = self.get_new_endpoint()
        try:
            conn = psycopg2.connect(
                host=endpoint,
                database='musicgen',
                user='musicgen_app',
                password=self.get_db_password()
            )
            cur = conn.cursor()
            
            # Test write
            cur.execute("""
                INSERT INTO disaster_recovery_log (event, timestamp)
                VALUES ('Regional failover completed', NOW())
            """)
            conn.commit()
            
            cur.close()
            conn.close()
            return True
        except Exception as e:
            print(f"Verification failed: {e}")
            return False
    
    def get_db_password(self):
        """Retrieve database password from Secrets Manager"""
        sm = boto3.client('secretsmanager', region_name=self.dr_region)
        response = sm.get_secret_value(SecretId='musicgen/database/password')
        return response['SecretString']

# Execute failover
if __name__ == "__main__":
    failover = RegionalDatabaseFailover('us-east-1', 'us-west-2')
    failover.execute_failover()
```

### Application Failover (60-120 minutes)

```bash
#!/bin/bash
# Application failover to DR region

# 1. Scale up DR Kubernetes cluster
eksctl scale nodegroup \
    --cluster musicgen-dr \
    --name workers \
    --nodes 20 \
    --nodes-min 10

# 2. Deploy full application stack
kubectl config use-context musicgen-dr
kubectl apply -f k8s/production/ --recursive

# 3. Sync models to DR region
aws s3 sync s3://musicgen-models-primary/ s3://musicgen-models-dr/ \
    --source-region us-east-1 \
    --region us-west-2

# 4. Update service mesh
kubectl apply -f istio/disaster-recovery/virtual-services.yaml

# 5. Verify all pods are running
kubectl wait --for=condition=ready pod -l app=musicgen -n production --timeout=300s
```

---

## Database Corruption

**Scenario**: Primary database is corrupted due to hardware failure, software bug, or malicious action.

### Detection and Assessment (0-15 minutes)

```sql
-- Check for corruption indicators
-- Run on primary database

-- 1. Check system catalogs
SELECT COUNT(*) FROM pg_class WHERE relkind = 'r';
SELECT COUNT(*) FROM pg_attribute WHERE attrelid > 0;

-- 2. Check for invalid indexes
SELECT 
    schemaname,
    tablename,
    indexname
FROM pg_indexes
WHERE NOT EXISTS (
    SELECT 1 FROM pg_class WHERE relname = indexname
);

-- 3. Run integrity checks
DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN SELECT tablename FROM pg_tables WHERE schemaname = 'public'
    LOOP
        BEGIN
            EXECUTE 'SELECT COUNT(*) FROM ' || quote_ident(r.tablename);
            RAISE NOTICE 'Table % is OK', r.tablename;
        EXCEPTION WHEN OTHERS THEN
            RAISE WARNING 'Table % is CORRUPTED: %', r.tablename, SQLERRM;
        END;
    END LOOP;
END $$;

-- 4. Check transaction ID wraparound
SELECT 
    datname,
    age(datfrozenxid) as xid_age,
    2^31 - age(datfrozenxid) as xids_remaining
FROM pg_database
ORDER BY age(datfrozenxid) DESC;
```

### Recovery Procedures

#### Option 1: Point-in-Time Recovery (Preferred)

```bash
#!/bin/bash
# Point-in-time recovery script

# 1. Identify corruption time
echo "Analyzing logs to find corruption time..."
CORRUPTION_TIME=$(psql -h postgres.musicgen.ai -U musicgen -c "
    SELECT MIN(timestamp) 
    FROM audit_log 
    WHERE event_type = 'data_anomaly'
" -t)

echo "Corruption detected at: $CORRUPTION_TIME"
RECOVERY_TIME=$(date -d "$CORRUPTION_TIME -5 minutes" +"%Y-%m-%d %H:%M:%S")

# 2. Create new instance from backup
aws rds restore-db-instance-to-point-in-time \
    --source-db-instance-identifier musicgen-primary \
    --target-db-instance-identifier musicgen-recovery \
    --restore-time "$RECOVERY_TIME" \
    --db-instance-class db.r6i.4xlarge \
    --multi-az

# 3. Wait for restoration
echo "Waiting for restoration to complete..."
aws rds wait db-instance-available \
    --db-instance-identifier musicgen-recovery

# 4. Verify data integrity
psql -h musicgen-recovery.xxx.rds.amazonaws.com -U musicgen << EOF
-- Verify critical tables
SELECT 'users', COUNT(*) FROM users
UNION ALL
SELECT 'generations', COUNT(*) FROM generations
UNION ALL
SELECT 'models', COUNT(*) FROM models;

-- Check for the corruption
SELECT COUNT(*) 
FROM information_schema.tables 
WHERE table_schema = 'public';
EOF

# 5. Replay missing transactions
python scripts/replay_transactions.py \
    --source postgres.musicgen.ai \
    --target musicgen-recovery.xxx.rds.amazonaws.com \
    --from-time "$RECOVERY_TIME" \
    --to-time "$(date +%Y-%m-%d %H:%M:%S)"
```

#### Option 2: Selective Table Recovery

```python
#!/usr/bin/env python3
"""
Selective table recovery for database corruption
"""
import psycopg2
import subprocess
from datetime import datetime

class SelectiveTableRecovery:
    def __init__(self, backup_host, production_host):
        self.backup_host = backup_host
        self.production_host = production_host
        
    def recover_table(self, schema, table, where_clause=None):
        """Recover specific table from backup"""
        print(f"Recovering table {schema}.{table}")
        
        # 1. Dump table from backup
        dump_file = f"/tmp/{table}_recovery.sql"
        dump_cmd = [
            'pg_dump',
            '-h', self.backup_host,
            '-U', 'musicgen',
            '-t', f'{schema}.{table}',
            '-f', dump_file,
            '--data-only' if where_clause else '--clean',
            'musicgen_prod'
        ]
        
        if where_clause:
            dump_cmd.extend(['--where', where_clause])
            
        subprocess.run(dump_cmd, check=True)
        
        # 2. Prepare production table
        prod_conn = psycopg2.connect(host=self.production_host, database='musicgen_prod')
        prod_cur = prod_conn.cursor()
        
        if not where_clause:
            # Full table replacement
            prod_cur.execute(f"TRUNCATE TABLE {schema}.{table} CASCADE")
        else:
            # Selective replacement
            prod_cur.execute(f"DELETE FROM {schema}.{table} WHERE {where_clause}")
        
        prod_conn.commit()
        
        # 3. Restore data
        restore_cmd = [
            'psql',
            '-h', self.production_host,
            '-U', 'musicgen',
            '-d', 'musicgen_prod',
            '-f', dump_file
        ]
        
        subprocess.run(restore_cmd, check=True)
        
        # 4. Verify restoration
        prod_cur.execute(f"SELECT COUNT(*) FROM {schema}.{table}")
        count = prod_cur.fetchone()[0]
        print(f"✅ Restored {count} rows to {schema}.{table}")
        
        prod_cur.close()
        prod_conn.close()
        
    def recover_corrupted_tables(self, corrupted_tables):
        """Recover multiple corrupted tables"""
        for table_info in corrupted_tables:
            try:
                self.recover_table(
                    table_info['schema'],
                    table_info['table'],
                    table_info.get('where_clause')
                )
            except Exception as e:
                print(f"❌ Failed to recover {table_info['table']}: {e}")

# Usage
if __name__ == "__main__":
    recovery = SelectiveTableRecovery(
        backup_host='backup.musicgen.ai',
        production_host='postgres.musicgen.ai'
    )
    
    # Example: Recover specific corrupted tables
    corrupted_tables = [
        {'schema': 'public', 'table': 'user_preferences'},
        {'schema': 'public', 'table': 'generation_metadata', 'where_clause': "created_at > '2024-01-15'"}
    ]
    
    recovery.recover_corrupted_tables(corrupted_tables)
```

---

## Ransomware Attack

**Scenario**: Systems are encrypted by ransomware, demanding payment for decryption.

### Immediate Response (0-30 minutes)

```bash
#!/bin/bash
# Ransomware incident response

# 1. ISOLATE - Prevent spread
echo "=== ISOLATING INFECTED SYSTEMS ==="

# Null route infected systems
for ip in $(cat /tmp/infected_ips.txt); do
    sudo ip route add blackhole $ip
done

# Block all outbound traffic except essential
iptables -P OUTPUT DROP
iptables -A OUTPUT -d 10.0.0.0/8 -j ACCEPT  # Internal only
iptables -A OUTPUT -p tcp --dport 443 -d security.musicgen.ai -j ACCEPT

# 2. PRESERVE EVIDENCE
echo "=== PRESERVING EVIDENCE ==="
mkdir -p /evidence/ransomware_$(date +%Y%m%d)

# Capture memory dump
for host in $(cat /tmp/infected_hosts.txt); do
    ssh $host "sudo dd if=/dev/mem of=/tmp/memory.dump"
    scp $host:/tmp/memory.dump /evidence/ransomware_$(date +%Y%m%d)/${host}_memory.dump
done

# Capture network traffic
tcpdump -i any -w /evidence/ransomware_$(date +%Y%m%d)/network.pcap &

# 3. IDENTIFY ENCRYPTION
echo "=== IDENTIFYING RANSOMWARE ==="

# Check for ransom notes
find / -name "*ransom*" -o -name "*decrypt*" -o -name "*.txt" -mtime -1 2>/dev/null

# Identify encrypted files
find /app /data -type f -exec file {} \; | grep -E "data|encrypted"

# Check for IoCs
grep -r "onion\|bitcoin\|monero" /var/log/
```

### Containment and Eradication (30 minutes - 2 hours)

```python
#!/usr/bin/env python3
"""
Ransomware containment and system verification
"""
import os
import hashlib
import json
from datetime import datetime
import boto3

class RansomwareResponse:
    def __init__(self):
        self.infected_systems = []
        self.clean_backups = []
        self.s3 = boto3.client('s3')
        
    def identify_clean_backups(self):
        """Find backups from before infection"""
        # Get infection time from indicators
        infection_time = self.get_infection_time()
        
        print(f"Infection detected at: {infection_time}")
        
        # List all backups
        response = self.s3.list_objects_v2(
            Bucket='musicgen-backups',
            Prefix='full-backup'
        )
        
        for obj in response.get('Contents', []):
            backup_time = obj['LastModified']
            
            if backup_time < infection_time:
                # Verify backup integrity
                if self.verify_backup_integrity(obj['Key']):
                    self.clean_backups.append({
                        'key': obj['Key'],
                        'time': backup_time,
                        'size': obj['Size']
                    })
        
        # Sort by time, newest first
        self.clean_backups.sort(key=lambda x: x['time'], reverse=True)
        
        print(f"Found {len(self.clean_backups)} clean backups")
        return self.clean_backups
    
    def verify_backup_integrity(self, backup_key):
        """Verify backup hasn't been tampered with"""
        # Download manifest
        manifest_key = backup_key.replace('.tar.gz', '.manifest')
        
        try:
            response = self.s3.get_object(Bucket='musicgen-backups', Key=manifest_key)
            manifest = json.loads(response['Body'].read())
            
            # Verify checksum
            stored_checksum = manifest.get('checksum')
            
            # Download and verify backup checksum
            response = self.s3.head_object(Bucket='musicgen-backups', Key=backup_key)
            etag = response['ETag'].strip('"')
            
            return etag == stored_checksum
        except Exception as e:
            print(f"Failed to verify {backup_key}: {e}")
            return False
    
    def get_infection_time(self):
        """Determine when infection started"""
        # Check various indicators
        indicators = []
        
        # Check file modification times
        for root, dirs, files in os.walk('/var/log'):
            for file in files:
                if 'ransom' in file.lower() or '.encrypted' in file:
                    filepath = os.path.join(root, file)
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    indicators.append(mtime)
        
        # Check system logs for suspicious activity
        # (Implementation depends on log format)
        
        if indicators:
            return min(indicators)
        else:
            # Default to 24 hours ago if no indicators
            return datetime.utcnow() - timedelta(hours=24)
    
    def clean_rebuild_plan(self):
        """Generate rebuild plan from clean backups"""
        if not self.clean_backups:
            print("❌ No clean backups found!")
            return None
            
        latest_clean = self.clean_backups[0]
        
        plan = {
            'backup_source': latest_clean['key'],
            'backup_time': latest_clean['time'].isoformat(),
            'recovery_steps': [
                'Provision new infrastructure in isolated VPC',
                'Restore from clean backup',
                'Apply security patches',
                'Implement additional monitoring',
                'Validate all systems',
                'Gradual traffic migration'
            ],
            'estimated_time': '4-6 hours',
            'data_loss': self.calculate_data_loss(latest_clean['time'])
        }
        
        return plan
    
    def calculate_data_loss(self, backup_time):
        """Calculate potential data loss"""
        current_time = datetime.utcnow()
        time_diff = current_time - backup_time
        
        hours_lost = time_diff.total_seconds() / 3600
        
        # Estimate based on typical usage
        estimated_loss = {
            'time_window': f"{hours_lost:.1f} hours",
            'estimated_generations': int(hours_lost * 100),  # 100 per hour average
            'estimated_users': int(hours_lost * 10),  # 10 new users per hour
            'recovery_method': 'Transaction log replay' if hours_lost < 24 else 'Partial recovery'
        }
        
        return estimated_loss

# Execute response
if __name__ == "__main__":
    response = RansomwareResponse()
    
    # Find clean backups
    clean_backups = response.identify_clean_backups()
    
    if clean_backups:
        print("\nClean backups available:")
        for backup in clean_backups[:5]:  # Show top 5
            print(f"  - {backup['key']} ({backup['time']})")
    
    # Generate recovery plan
    plan = response.clean_rebuild_plan()
    
    if plan:
        print("\nRecovery Plan:")
        print(json.dumps(plan, indent=2, default=str))
```

### Recovery from Clean Backup (2-6 hours)

```bash
#!/bin/bash
# Ransomware recovery from clean infrastructure

# 1. Provision new isolated environment
cd infrastructure/terraform
terraform workspace new ransomware-recovery
terraform apply -var-file=ransomware-recovery.tfvars -auto-approve

# 2. Restore from verified clean backup
CLEAN_BACKUP="s3://musicgen-backups/full-backup-20240114-2300.tar.gz"

# Download and extract
aws s3 cp $CLEAN_BACKUP /tmp/clean_backup.tar.gz
tar -xzf /tmp/clean_backup.tar.gz -C /recovery/

# 3. Restore database
psql -h new-postgres.musicgen.ai -U musicgen < /recovery/database/backup.sql

# 4. Restore application data
aws s3 sync /recovery/models/ s3://musicgen-models-recovery/
aws s3 sync /recovery/audio/ s3://musicgen-audio-recovery/

# 5. Deploy applications with additional security
kubectl apply -f k8s/ransomware-recovery/

# 6. Implement additional security measures
# - Enable application sandboxing
# - Implement strict egress rules
# - Enable real-time file monitoring
# - Deploy EDR solution

# 7. Validate recovery
./scripts/full_system_validation.sh --strict
```

---

## Critical Data Loss

**Scenario**: Critical data is lost due to accidental deletion, corruption, or sabotage.

### Data Recovery Matrix

| Data Type | Recovery Method | RPO | Recovery Time |
|-----------|----------------|-----|---------------|
| User Data | Database PITR | 1 hour | 30 minutes |
| Generated Audio | S3 Versioning | 24 hours | 1-2 hours |
| Model Weights | S3 Glacier | 7 days | 4-6 hours |
| Configuration | Git History | Real-time | 15 minutes |
| Logs | CloudWatch/S3 | 1 hour | 30 minutes |

### Recovery Procedures

```python
#!/usr/bin/env python3
"""
Critical data recovery orchestration
"""
import boto3
import psycopg2
from datetime import datetime, timedelta
import concurrent.futures

class DataRecoveryOrchestrator:
    def __init__(self):
        self.s3 = boto3.client('s3')
        self.rds = boto3.client('rds')
        
    def recover_user_data(self, target_time):
        """Recover user data to specific point in time"""
        print(f"Recovering user data to {target_time}")
        
        # Create new database from PITR
        response = self.rds.restore_db_instance_to_point_in-time(
            SourceDBInstanceIdentifier='musicgen-primary',
            TargetDBInstanceIdentifier=f'recovery-{int(datetime.utcnow().timestamp())}',
            RestoreTime=target_time,
            UseLatestRestorableTime=False,
            DBInstanceClass='db.r6i.2xlarge'
        )
        
        recovery_instance = response['DBInstance']['DBInstanceIdentifier']
        
        # Wait for restoration
        waiter = self.rds.get_waiter('db_instance_available')
        waiter.wait(DBInstanceIdentifier=recovery_instance)
        
        return recovery_instance
    
    def recover_s3_objects(self, bucket, prefix, target_time):
        """Recover deleted S3 objects"""
        print(f"Recovering S3 objects from {bucket}/{prefix}")
        
        recovered_count = 0
        
        # List object versions
        paginator = self.s3.get_paginator('list_object_versions')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            for version in page.get('Versions', []):
                if version['LastModified'] <= target_time:
                    # Check if latest version is delete marker
                    if self.is_deleted(bucket, version['Key']):
                        # Restore by removing delete marker
                        self.s3.delete_object(
                            Bucket=bucket,
                            Key=version['Key'],
                            VersionId=self.get_delete_marker_version(bucket, version['Key'])
                        )
                        recovered_count += 1
                        print(f"Recovered: {version['Key']}")
        
        return recovered_count
    
    def is_deleted(self, bucket, key):
        """Check if object has delete marker"""
        try:
            response = self.s3.head_object(Bucket=bucket, Key=key)
            return False
        except self.s3.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                # Check for delete marker
                versions = self.s3.list_object_versions(
                    Bucket=bucket,
                    Prefix=key,
                    MaxKeys=1
                )
                return bool(versions.get('DeleteMarkers'))
        return False
    
    def parallel_recovery(self, recovery_tasks):
        """Execute multiple recovery tasks in parallel"""
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_task = {}
            
            for task in recovery_tasks:
                if task['type'] == 'database':
                    future = executor.submit(
                        self.recover_user_data,
                        task['target_time']
                    )
                elif task['type'] == 's3':
                    future = executor.submit(
                        self.recover_s3_objects,
                        task['bucket'],
                        task['prefix'],
                        task['target_time']
                    )
                
                future_to_task[future] = task
            
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results[task['name']] = {
                        'status': 'success',
                        'result': result
                    }
                except Exception as e:
                    results[task['name']] = {
                        'status': 'failed',
                        'error': str(e)
                    }
        
        return results

# Execute recovery
if __name__ == "__main__":
    orchestrator = DataRecoveryOrchestrator()
    
    # Define recovery tasks
    target_time = datetime.utcnow() - timedelta(hours=2)
    
    recovery_tasks = [
        {
            'name': 'user_database',
            'type': 'database',
            'target_time': target_time
        },
        {
            'name': 'generated_audio',
            'type': 's3',
            'bucket': 'musicgen-audio',
            'prefix': 'generations/',
            'target_time': target_time
        },
        {
            'name': 'user_uploads',
            'type': 's3',
            'bucket': 'musicgen-uploads',
            'prefix': 'user-content/',
            'target_time': target_time
        }
    ]
    
    # Execute parallel recovery
    results = orchestrator.parallel_recovery(recovery_tasks)
    
    # Report results
    print("\nRecovery Results:")
    for task_name, result in results.items():
        status = "✅" if result['status'] == 'success' else "❌"
        print(f"{status} {task_name}: {result}")
```

---

## Regional Disaster

**Scenario**: Entire AWS region is unavailable due to natural disaster or major infrastructure failure.

### Multi-Region Failover

```bash
#!/bin/bash
# Multi-region disaster recovery

# Configuration
PRIMARY_REGION="us-east-1"
DR_REGIONS=("us-west-2" "eu-west-1" "ap-southeast-1")
RECOVERY_REGION=""

# 1. Determine best recovery region
echo "=== Selecting Recovery Region ==="
for region in "${DR_REGIONS[@]}"; do
    echo "Testing region: $region"
    
    # Test region availability
    if aws ec2 describe-regions --region $region &>/dev/null; then
        # Check infrastructure readiness
        ready=$(aws cloudformation describe-stacks \
            --region $region \
            --stack-name musicgen-dr-infrastructure \
            --query 'Stacks[0].StackStatus' \
            --output text 2>/dev/null)
        
        if [ "$ready" == "CREATE_COMPLETE" ] || [ "$ready" == "UPDATE_COMPLETE" ]; then
            RECOVERY_REGION=$region
            echo "✅ Selected recovery region: $RECOVERY_REGION"
            break
        fi
    fi
done

if [ -z "$RECOVERY_REGION" ]; then
    echo "❌ No suitable recovery region found!"
    exit 1
fi

# 2. Activate recovery region
echo "=== Activating Recovery Region: $RECOVERY_REGION ==="

# Update Route53 for geo-routing
aws route53 change-resource-record-sets \
    --hosted-zone-id Z1234567890ABC \
    --change-batch file://route53-disaster-recovery.json

# 3. Scale up recovery infrastructure
aws autoscaling set-desired-capacity \
    --region $RECOVERY_REGION \
    --auto-scaling-group-name musicgen-dr-api-asg \
    --desired-capacity 20

# 4. Restore data from cross-region replicas
echo "=== Restoring Data ==="

# Promote read replica to primary
aws rds promote-read-replica \
    --region $RECOVERY_REGION \
    --db-instance-identifier musicgen-dr-replica

# Sync S3 data (if needed)
aws s3 sync \
    s3://musicgen-primary-$PRIMARY_REGION/ \
    s3://musicgen-dr-$RECOVERY_REGION/ \
    --source-region $PRIMARY_REGION \
    --region $RECOVERY_REGION

# 5. Deploy full application stack
kubectl config use-context $RECOVERY_REGION
kubectl apply -f k8s/disaster-recovery/region-$RECOVERY_REGION/

# 6. Update CDN origin
aws cloudfront get-distribution-config --id E1234567890ABC > /tmp/cf-config.json
# Update origin in config
jq '.DistributionConfig.Origins.Items[0].DomainName = "'$RECOVERY_REGION'.musicgen.ai"' /tmp/cf-config.json > /tmp/cf-config-updated.json
# Apply update
aws cloudfront update-distribution --id E1234567890ABC --distribution-config file:///tmp/cf-config-updated.json

# 7. Verify recovery
echo "=== Verifying Recovery ==="
./scripts/verify_regional_recovery.sh $RECOVERY_REGION
```

---

## Recovery Validation Checklist

After any disaster recovery:

- [ ] All services responding to health checks
- [ ] Database integrity verified
- [ ] No data corruption detected
- [ ] Authentication working
- [ ] API endpoints functional
- [ ] Generation pipeline operational
- [ ] Monitoring and alerting active
- [ ] Backup processes resumed
- [ ] Security measures in place
- [ ] Performance within SLA
- [ ] Customer communication sent
- [ ] Incident report drafted
- [ ] Lessons learned documented
- [ ] Recovery procedures updated

---

## Contact Information

### Crisis Team

| Role | Name | Phone | Email |
|------|------|-------|-------|
| Incident Commander | John Smith | +1-555-0001 | john.smith@musicgen.ai |
| Technical Lead | Jane Doe | +1-555-0002 | jane.doe@musicgen.ai |
| Database Admin | Bob Wilson | +1-555-0003 | bob.wilson@musicgen.ai |
| Security Lead | Alice Brown | +1-555-0004 | alice.brown@musicgen.ai |
| Communications | Tom Davis | +1-555-0005 | tom.davis@musicgen.ai |

### External Contacts

- **AWS Support**: 1-800-xxx-xxxx (Enterprise Support)
- **Datadog Support**: support@datadoghq.com
- **PagerDuty**: 1-844-xxx-xxxx
- **Legal Counsel**: legal@lawfirm.com
- **PR Agency**: pr@agency.com

---

**Remember**: In a disaster, prioritize:
1. **Safety** - Ensure team safety first
2. **Communication** - Keep stakeholders informed
3. **Data Integrity** - Preserve data over availability
4. **Documentation** - Document all actions taken
5. **Learning** - Capture lessons for improvement