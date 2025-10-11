# Preprocessor API Implementation Summary

This document summarizes the implementation of the preprocessor API functions for the Apex application.

## Implementation Overview

The preprocessor API provides a complete interface for:
1. Listing available preprocessors
2. Downloading preprocessor models
3. Running preprocessors on media files
4. Real-time job tracking via WebSocket
5. Retrieving job results

## Files Created/Modified

### Main Process
- **Modified**: `apex/packages/main/src/modules/ApexApi.ts`
  - Added `registerPreprocessorHandlers()` method
  - Implemented 6 IPC handlers for preprocessor operations
  - Added WebSocket connection management
  - Handlers: list, get, download, run, status, result
  - WebSocket handlers: connect-ws, disconnect-ws

- **Modified**: `apex/packages/main/package.json`
  - Added dependency: `ws@^8.18.0`
  - Added dev dependency: `@types/ws@^8.5.10`

### Preload Layer
- **Modified**: `apex/packages/preload/src/index.ts`
  - Added 9 new exported functions:
    - `listPreprocessors()`
    - `getPreprocessor()`
    - `downloadPreprocessor()`
    - `runPreprocessor()`
    - `getPreprocessorStatus()`
    - `getPreprocessorResult()`
    - `connectPreprocessorWebSocket()`
    - `disconnectPreprocessorWebSocket()`
    - Event listeners: `onPreprocessorWebSocketUpdate/Status/Error()`

### Renderer Layer
- **Created**: `apex/packages/renderer/src/lib/preprocessor/api.ts`
  - Complete typed API wrapper functions
  - `PreprocessorJob` helper class for managing jobs
  - TypeScript interfaces for all data types
  - Comprehensive JSDoc documentation

- **Created**: `apex/packages/renderer/src/lib/preprocessor/index.ts`
  - Central export point for preprocessor API
  - Exports all functions and types

- **Created**: `apex/packages/renderer/src/components/preprocessor/PreprocessorExample.tsx`
  - Complete React component demonstrating usage
  - Shows listing, downloading, running, and tracking
  - Real-time progress updates

### Documentation
- **Created**: `apex/PREPROCESSOR_API_USAGE.md`
  - Comprehensive usage guide
  - Code examples for all functions
  - React hook example
  - Troubleshooting section

- **Created**: `apex/PREPROCESSOR_API_SUMMARY.md` (this file)
  - Implementation overview
  - File structure summary

## API Endpoints Mapped

### Backend → Main Process → Preload → Renderer

1. **GET /preprocessor/list**
   - Main: `preprocessor:list`
   - Preload: `listPreprocessors()`
   - Renderer: `listPreprocessors()`

2. **GET /preprocessor/get/{name}**
   - Main: `preprocessor:get`
   - Preload: `getPreprocessor()`
   - Renderer: `getPreprocessor()`

3. **POST /preprocessor/download**
   - Main: `preprocessor:download`
   - Preload: `downloadPreprocessor()`
   - Renderer: `downloadPreprocessor()`

4. **POST /preprocessor/run**
   - Main: `preprocessor:run`
   - Preload: `runPreprocessor()`
   - Renderer: `runPreprocessor()`

5. **GET /preprocessor/status/{job_id}**
   - Main: `preprocessor:status`
   - Preload: `getPreprocessorStatus()`
   - Renderer: `getPreprocessorStatus()`

6. **GET /preprocessor/result/{job_id}**
   - Main: `preprocessor:result`
   - Preload: `getPreprocessorResult()`
   - Renderer: `getPreprocessorResult()`

7. **WS /ws/job/{job_id}**
   - Main: `preprocessor:connect-ws`, `preprocessor:disconnect-ws`
   - Preload: `connectPreprocessorWebSocket()`, `disconnectPreprocessorWebSocket()`
   - Renderer: `connectJobWebSocket()`, `disconnectJobWebSocket()`

## Key Features

### Type Safety
- Full TypeScript support throughout the stack
- Interfaces for all request/response types
- Generic `ConfigResponse<T>` wrapper for all API calls

### WebSocket Support
- Real-time job updates
- Automatic reconnection handling
- Multiple simultaneous connections
- Event-based subscription model

### Helper Classes
- `PreprocessorJob` class for easy job management
- Automatic cleanup on disconnect
- Multiple event subscriptions
- Unified interface for status and result queries

### Error Handling
- Consistent error response format
- Network error handling
- WebSocket error events
- Connection status tracking

## Usage Patterns

### Simple List
```typescript
const result = await listPreprocessors();
if (result.success) {
  console.log(result.data.preprocessors);
}
```

### Download with Progress
```typescript
const result = await downloadPreprocessor('model_name');
const job = new PreprocessorJob(result.data.job_id);
job.onUpdate(data => console.log(`Progress: ${data.progress * 100}%`));
await job.connect();
```

### Run with Real-time Updates
```typescript
const result = await runPreprocessor({
  preprocessor_name: 'depth_anything_v2',
  input_path: '/path/to/file.mp4'
});
const job = new PreprocessorJob(result.data.job_id);
job.onUpdate(data => {
  if (data.status === 'complete') {
    console.log('Done!', data.result_path);
  }
});
await job.connect();
```

## Testing

To test the implementation:

1. **Install dependencies**:
   ```bash
   cd apex
   npm install
   ```

2. **Start the backend**:
   ```bash
   cd apex-engine
   uvicorn src.api.main:app --reload
   ```

3. **Use the example component**:
   - Import `PreprocessorExample` in your app
   - Navigate to the component
   - Test listing, downloading, and running

## Next Steps

Potential enhancements:
- Add job cancellation support
- Implement job queue management UI
- Add batch processing capabilities
- Implement result caching and preview
- Add preprocessor parameter validation UI
- Create reusable React hooks for common patterns

## Dependencies

### Runtime
- `ws@^8.18.0` - WebSocket client for Node.js

### Development
- `@types/ws@^8.5.10` - TypeScript definitions for ws

## Notes

- WebSocket connections are managed in the main process for security
- All preprocessor operations are asynchronous
- Job IDs can be persisted and used across app restarts
- Results are cached on the backend for retrieval
- The API supports both HTTP polling and WebSocket streaming for updates

