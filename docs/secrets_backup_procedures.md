# Secrets Backup and Recovery Procedures

This document outlines comprehensive procedures for backing up, storing, and recovering secrets in the Music Gen AI system.

## Overview

The secrets backup system ensures business continuity by maintaining secure, recoverable copies of all critical secrets while adhering to security best practices and compliance requirements.

## Backup Strategy

### Backup Types

#### 1. Metadata-Only Backups (Default)
- **Content**: Secret names, types, expiration dates, rotation schedules
- **Excludes**: Actual secret values
- **Use Case**: Disaster recovery planning, system documentation
- **Security**: Low risk, can be stored in standard backup systems
- **Frequency**: Daily

#### 2. Full Value Backups (Encrypted)
- **Content**: Complete secret values and metadata
- **Security**: Encrypted with master encryption key
- **Use Case**: Complete system recovery, secret value restoration
- **Frequency**: Weekly for critical secrets, monthly for others
- **Retention**: 1 year minimum

### Backup Locations

#### Primary Backup Storage
- **Location**: HashiCorp Vault (Production)
- **Encryption**: Vault native encryption + transit encryption
- **Access Control**: Restricted service accounts only
- **Replication**: Multi-region with automatic failover

#### Secondary Backup Storage
- **Location**: Encrypted cloud storage (AWS S3/Azure Blob)
- **Encryption**: Client-side encryption before upload
- **Access Control**: IAM policies with MFA requirements
- **Versioning**: Enabled with lifecycle policies

#### Emergency Backup Storage
- **Location**: Offline encrypted storage
- **Encryption**: Hardware security module (HSM) protected
- **Access Control**: Physical security + multi-person access
- **Update Frequency**: Monthly

## Backup Procedures

### Automated Daily Backup

```bash
# Automated backup script (runs via cron)
#!/bin/bash

# Daily metadata-only backup
python scripts/secrets_cli.py backup \
    --output "/backup/daily/secrets_metadata_$(date +%Y%m%d).json" \
    --metadata-only

# Upload to cloud storage with encryption
aws s3 cp "/backup/daily/secrets_metadata_$(date +%Y%m%d).json" \
    "s3://musicgen-secrets-backup/daily/" \
    --sse-kms-key-id "arn:aws:kms:us-east-1:account:key/key-id"
```

### Weekly Full Backup

```bash
# Weekly encrypted full backup
python scripts/secrets_cli.py backup \
    --output "/backup/weekly/secrets_full_$(date +%Y%m%d).json" \
    --include-values \
    --encrypt-backup \
    --force

# Verify backup integrity
python scripts/secrets_cli.py verify-backup \
    "/backup/weekly/secrets_full_$(date +%Y%m%d).json"
```

### Manual Backup Commands

#### Create Metadata Backup
```bash
# Basic metadata backup
python scripts/secrets_cli.py backup secrets_backup.json

# Include specific metadata
python scripts/secrets_cli.py backup secrets_backup.json \
    --include-metadata \
    --include-expiration \
    --include-rotation-schedule
```

#### Create Full Value Backup
```bash
# Full backup with encryption (requires confirmation)
python scripts/secrets_cli.py backup secrets_full_backup.json \
    --include-values \
    --encrypt-backup

# Emergency full backup (skips confirmations)
python scripts/secrets_cli.py backup emergency_backup.json \
    --include-values \
    --encrypt-backup \
    --force
```

#### Backup Specific Secret Categories
```bash
# Backup only critical secrets
python scripts/secrets_cli.py backup critical_secrets.json \
    --filter-tag "priority=critical" \
    --include-values

# Backup database-related secrets
python scripts/secrets_cli.py backup db_secrets.json \
    --filter-tag "category=database" \
    --include-values
```

## Backup Encryption

### Encryption Keys

#### Master Backup Encryption Key
- **Purpose**: Encrypts all backup files
- **Algorithm**: AES-256-GCM
- **Key Derivation**: PBKDF2 with 100,000 iterations
- **Storage**: Hardware Security Module (HSM)
- **Rotation**: Annually with key versioning

#### Backup Key Management
```bash
# Generate new backup encryption key
python scripts/secrets_cli.py generate-backup-key \
    --output backup_key.key \
    --algorithm AES256

# Rotate backup encryption key
python scripts/secrets_cli.py rotate-backup-key \
    --current-key backup_key.key \
    --new-key backup_key_new.key \
    --re-encrypt-backups
```

### Encryption Implementation

#### Client-Side Encryption
```python
# Backup encryption implementation
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import os
import base64

def encrypt_backup(data: str, password: str) -> bytes:
    """Encrypt backup data with password-derived key."""
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    fernet = Fernet(key)
    
    encrypted_data = fernet.encrypt(data.encode())
    return salt + encrypted_data
```

## Recovery Procedures

### Pre-Recovery Validation

#### 1. Backup Integrity Check
```bash
# Verify backup file integrity
python scripts/secrets_cli.py verify-backup backup_file.json

# Check backup encryption status
python scripts/secrets_cli.py backup-info backup_file.json
```

#### 2. Environment Preparation
```bash
# Ensure clean secrets environment
python scripts/secrets_cli.py list --verify-clean

# Backup current state before recovery
python scripts/secrets_cli.py backup pre_recovery_backup.json
```

### Recovery Scenarios

#### Scenario 1: Single Secret Recovery
```bash
# Restore specific secret from backup
python scripts/secrets_cli.py restore-secret \
    --backup-file weekly_backup.json \
    --secret-name JWT_SECRET_KEY \
    --confirm-overwrite

# Verify secret restoration
python scripts/secrets_cli.py get JWT_SECRET_KEY --show-value
```

#### Scenario 2: Category Recovery
```bash
# Restore all database secrets
python scripts/secrets_cli.py restore \
    --backup-file full_backup.json \
    --filter-tag "category=database" \
    --overwrite-existing

# Validate restored secrets
python scripts/secrets_cli.py validate --secrets DATABASE_PASSWORD REDIS_PASSWORD
```

#### Scenario 3: Full System Recovery
```bash
# Complete system recovery from backup
python scripts/secrets_cli.py restore \
    --backup-file emergency_backup.json \
    --full-restore \
    --overwrite-existing \
    --verify-after-restore

# Post-recovery validation
python scripts/secrets_cli.py validate --production
python scripts/secrets_cli.py health-advanced
```

### Disaster Recovery

#### Emergency Recovery Process
1. **Assess Scope**: Determine which secrets are compromised/lost
2. **Secure Environment**: Ensure recovery environment is secure
3. **Locate Backups**: Identify most recent valid backup
4. **Decrypt and Restore**: Follow full recovery procedure
5. **Validate System**: Comprehensive testing of restored secrets
6. **Rotate Compromised**: Immediately rotate any potentially compromised secrets

#### Recovery Time Objectives (RTO)
- **Single Secret**: 15 minutes
- **Category Recovery**: 30 minutes
- **Full System Recovery**: 2 hours
- **Emergency Recovery**: 4 hours (including validation)

## Backup Monitoring and Alerting

### Automated Monitoring

#### Backup Health Checks
```bash
# Daily backup verification
python scripts/secrets_cli.py backup-health-check \
    --check-integrity \
    --check-encryption \
    --check-accessibility

# Alert on backup failures
python scripts/secrets_cli.py backup-alerts \
    --email alerts@musicgen.ai \
    --threshold-age 48h
```

#### Key Metrics
- **Backup Success Rate**: >99.9%
- **Backup Integrity**: 100% verified
- **Recovery Test Success**: Monthly validation
- **Encryption Key Health**: Quarterly rotation

### Alerting Configuration

#### Critical Alerts
- Backup failure for >24 hours
- Backup corruption detected
- Encryption key rotation failure
- Unauthorized backup access

#### Warning Alerts
- Backup size anomalies
- Backup performance degradation
- Encryption key nearing expiration

## Compliance and Auditing

### Audit Trail
- All backup operations logged with timestamps
- Access logs for backup files
- Encryption key usage tracking
- Recovery operation documentation

### Compliance Requirements
- **SOC 2**: Documented backup and recovery procedures
- **ISO 27001**: Information security management compliance
- **GDPR**: Data protection and recovery capabilities
- **PCI DSS**: Secure backup of payment-related secrets

### Audit Commands
```bash
# Generate backup audit report
python scripts/secrets_cli.py backup-audit-report \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --output backup_audit_2024.json

# Verify compliance status
python scripts/secrets_cli.py compliance-check \
    --framework SOC2 \
    --backup-requirements
```

## Testing and Validation

### Regular Testing Schedule

#### Monthly Recovery Tests
- Test single secret recovery
- Validate backup integrity
- Verify encryption/decryption

#### Quarterly Disaster Recovery Drills
- Full system recovery simulation
- Multi-region failover testing
- Performance and RTO validation

#### Annual Security Reviews
- Backup encryption assessment
- Access control validation
- Compliance audit

### Testing Commands
```bash
# Automated recovery test
python scripts/secrets_cli.py test-recovery \
    --backup-file test_backup.json \
    --dry-run \
    --report test_recovery_report.json

# Performance testing
python scripts/secrets_cli.py benchmark-backup \
    --secret-count 1000 \
    --measure-encryption-time \
    --measure-upload-time
```

## Security Considerations

### Access Control
- **Backup Files**: Encrypted at rest, restricted access
- **Encryption Keys**: HSM-protected, multi-person access
- **Recovery Operations**: Logged and audited
- **Cloud Storage**: IAM policies with MFA

### Best Practices
1. Never store backup encryption keys with backup files
2. Regular validation of backup integrity
3. Secure deletion of temporary backup files
4. Multi-region backup distribution
5. Regular security assessments

### Incident Response
- Immediate backup isolation if compromise suspected
- Emergency key rotation procedures
- Forensic analysis of backup access logs
- Communication protocols for stakeholders

## Troubleshooting

### Common Issues

#### Backup Failures
```bash
# Check backup permissions
python scripts/secrets_cli.py check-backup-permissions

# Verify storage connectivity
python scripts/secrets_cli.py test-backup-storage

# Diagnose encryption issues
python scripts/secrets_cli.py diagnose-backup-encryption
```

#### Recovery Issues
```bash
# Validate backup file format
python scripts/secrets_cli.py validate-backup-format backup.json

# Test decryption without restoring
python scripts/secrets_cli.py test-decrypt backup.json

# Check environment compatibility
python scripts/secrets_cli.py check-recovery-environment
```

### Emergency Contacts
- **Primary**: DevOps Team (24/7)
- **Secondary**: Security Team
- **Escalation**: CTO/Security Officer

## References
- [NIST SP 800-57: Key Management](https://csrc.nist.gov/publications/detail/sp/800-57-part-1/rev-5/final)
- [OWASP Secrets Management](https://owasp.org/www-project-secrets-management/)
- [HashiCorp Vault Backup Guide](https://learn.hashicorp.com/vault)