# Logging Infrastructure Fix Summary

## Issues Fixed

1. **Missing Function Parameters**
   - `setup_logging()` now accepts `level` and `log_file` parameters
   - Added fallback to config values if parameters not provided

2. **Missing Classes and Functions**
   - Implemented `LoggerMixin` class that provides a cached logger property
   - Implemented `log_function_call` decorator for timing function execution
   - Implemented `log_gpu_memory` function for GPU memory tracking

3. **Logger Hierarchy Issues**
   - Fixed logger propagation in tests by using consistent naming
   - Ensured `music_gen` logger is properly configured with handlers

## Key Changes Made

### `/src/musicgen/infrastructure/monitoring/logging.py`

1. **Updated Imports**
   ```python
   import time
   import functools
   from typing import Optional, Callable
   ```

2. **Enhanced setup_logging**
   - Now accepts parameters: `level` and `log_file`
   - Fallback to config module if available

3. **Added LoggerMixin Class**
   - Provides cached logger instances per class
   - Automatically uses module + class name for logger

4. **Added log_function_call Decorator**
   - Logs function entry and exit with timing
   - Handles exceptions properly
   - Uses functools.wraps to preserve function metadata

5. **Added log_gpu_memory Function**
   - Checks for PyTorch and CUDA availability
   - Reports GPU memory usage in GB
   - Silent when GPU not available

## Test Results

All 13 logging tests now pass:
- ✅ LoggerMixin functionality
- ✅ Function call logging with timing
- ✅ GPU memory logging
- ✅ Logging setup with various configurations
- ✅ File handler creation
- ✅ Exception logging
- ✅ Log aggregation from multiple sources

## Usage Examples

```python
# Using LoggerMixin
class MyService(LoggerMixin):
    def process(self):
        self.logger.info("Processing started")

# Using function decorator
@log_function_call
def expensive_operation():
    # Function will be timed automatically
    pass

# Using GPU memory logging
logger = get_logger("my_module")
log_gpu_memory(logger, "model loading")
```

The logging infrastructure is now fully functional and professional-grade, supporting structured logging, timing decorators, and GPU monitoring.