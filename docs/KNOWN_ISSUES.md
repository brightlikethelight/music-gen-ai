# Known Issues and Technical Debt

> **Academic Project**: Harvard CS 109B — AI Music Generation

## Current Status (March 2026)

### CI/CD Pipeline
**Status**: Defined, all checks pass locally (first remote run pending push)
- Code quality: black, isort, flake8, mypy — all clean
- Test gate: `ci-pass` job gates on all checks
- Matrix: Python 3.10, 3.11, 3.12

### Test Suite
- **452+ tests passing**, 0 failing, ~20 skipped (model-dependent)
- **92% coverage** (threshold: 75%)
- Skipped tests require GPU or real model weights

### Remaining Limitations

| Area | Status | Notes |
|------|--------|-------|
| State persistence | In-memory only | All data lost on restart; no database backend |
| Stub endpoints | `/audio/analyze`, `/audio/waveform`, `/search` | Return hardcoded data |
| `/health/services` | Fabricated response times | Not actually probing services |
| WebSocket streaming | Not implemented | Clients must poll `/status/{job_id}` |
| Rate limiting | IP-only | Per-user rate limiting not connected to auth tiers |
| Redis auth | Sync client | Blocks event loop; needs `redis.asyncio` |

### Security Notes
- JWT auth fully implemented with refresh tokens
- CORS restricted to configured origins (fail-closed in production)
- Symlink protection on audio file serving
- Short JWT keys rejected in production
- Prompt validation on generation endpoints

---

*Last Updated: March 2026*
*Course: Harvard CS 109B — Advanced Data Science*
