# Preprocessor API Usage Guide

This document explains how to use the Apex preprocessor API functions in the renderer to list, download, run, and track preprocessing jobs.

## Overview

The preprocessor API provides:
- List all available preprocessors with metadata
- Download preprocessor models
- Run preprocessors on media files
- Track job progress in real-time via WebSocket
- Get job results

## Architecture

The implementation follows Electron's IPC pattern with WebSocket support:

1. **Backend API** (`apex-engine/src/api/preprocessor.py`): FastAPI endpoints with Ray for distributed processing
2. **Main Process** (`apex/packages/main/src/modules/ApexApi.ts`): IPC handlers and WebSocket bridge
3. **Preload Script** (`apex/packages/preload/src/index.ts`): Exposes IPC functions via contextBridge
4. **Renderer Helpers** (`apex/packages/renderer/src/lib/preprocessor/`): Typed wrapper functions and helper classes

## Installation

Before using the preprocessor API, install the required dependencies:

```bash
cd apex
npm install
```

This will install `ws` and `@types/ws` packages needed for WebSocket support.

## Available Functions

### List All Preprocessors

```typescript
import { listPreprocessors } from '@/lib/preprocessor';

const result = await listPreprocessors(true); // true = check download status
if (result.success && result.data) {
  console.log(`Found ${result.data.count} preprocessors`);
  result.data.preprocessors.forEach(prep => {
    console.log(`${prep.name}: ${prep.is_downloaded ? 'Downloaded' : 'Not Downloaded'}`);
  });
}
```

### Get Preprocessor Details

```typescript
import { getPreprocessor } from '@/lib/preprocessor';

const result = await getPreprocessor('depth_anything_v2');
if (result.success && result.data) {
  console.log('Name:', result.data.name);
  console.log('Description:', result.data.description);
  console.log('Parameters:', result.data.parameters);
}
```

### Download a Preprocessor

```typescript
import { downloadPreprocessor, PreprocessorJob } from '@/lib/preprocessor';

// Start download
const result = await downloadPreprocessor('depth_anything_v2');
if (result.success && result.data) {
  const job = new PreprocessorJob(result.data.job_id);
  
  // Subscribe to updates
  job.onUpdate((data) => {
    if (data.progress !== undefined) {
      console.log(`Download progress: ${Math.round(data.progress * 100)}%`);
    }
    if (data.status === 'complete') {
      console.log('Download complete!');
      job.disconnect();
    }
  });
  
  job.onError((data) => {
    console.error('Download error:', data.error);
  });
  
  // Connect to WebSocket
  await job.connect();
}
```

### Run a Preprocessor

```typescript
import { runPreprocessor, PreprocessorJob } from '@/lib/preprocessor';

// Run preprocessor
const result = await runPreprocessor({
  preprocessor_name: 'depth_anything_v2',
  input_path: '/path/to/video.mp4',
  download_if_needed: true,
  params: {
    // Optional parameters specific to the preprocessor
  },
  start_frame: 0,     // Optional: for video processing
  end_frame: 100,     // Optional: for video processing
});

if (result.success && result.data) {
  const job = new PreprocessorJob(result.data.job_id);
  
  // Subscribe to updates
  job.onUpdate((data) => {
    if (data.progress !== undefined) {
      console.log(`Processing: ${Math.round(data.progress * 100)}%`);
    }
    if (data.status === 'complete') {
      console.log('Processing complete!');
      console.log('Result path:', data.result_path);
      job.disconnect();
    }
    if (data.status === 'error') {
      console.error('Processing error:', data.error);
      job.disconnect();
    }
  });
  
  job.onStatus((data) => {
    console.log('Connection status:', data.status);
  });
  
  job.onError((data) => {
    console.error('WebSocket error:', data.error);
  });
  
  // Connect to WebSocket
  await job.connect();
}
```

### Get Job Status (Polling Alternative)

```typescript
import { getPreprocessorStatus } from '@/lib/preprocessor';

const result = await getPreprocessorStatus(jobId);
if (result.success && result.data) {
  console.log('Status:', result.data.status);
  console.log('Message:', result.data.message);
}
```

### Get Job Result

```typescript
import { getPreprocessorResult } from '@/lib/preprocessor';

const result = await getPreprocessorResult(jobId);
if (result.success && result.data) {
  if (result.data.status === 'complete') {
    console.log('Result path:', result.data.result_path);
    console.log('Type:', result.data.type);
    console.log('Preprocessor:', result.data.preprocessor);
  } else if (result.data.status === 'error') {
    console.error('Job failed:', result.data.error);
  }
}
```

## Using the PreprocessorJob Helper Class

The `PreprocessorJob` class provides a convenient way to manage jobs with automatic WebSocket tracking:

```typescript
import { PreprocessorJob, runPreprocessor } from '@/lib/preprocessor';

async function processVideo(videoPath: string) {
  // Start the job
  const result = await runPreprocessor({
    preprocessor_name: 'depth_anything_v2',
    input_path: videoPath,
  });
  
  if (!result.success) {
    throw new Error(result.error);
  }
  
  const job = new PreprocessorJob(result.data!.job_id);
  
  // Set up event handlers
  job.onUpdate((data) => {
    console.log('Update:', data);
  });
  
  job.onStatus((data) => {
    console.log('Connection:', data.status);
  });
  
  job.onError((data) => {
    console.error('Error:', data.error);
  });
  
  // Connect and wait
  await job.connect();
  
  // You can also poll status manually if needed
  const status = await job.getStatus();
  console.log('Current status:', status);
  
  // When done, disconnect
  await job.disconnect();
}
```

## WebSocket Events

When connected via WebSocket, you'll receive updates with the following structure:

### Update Events
```typescript
{
  status: 'queued' | 'downloading' | 'processing' | 'complete' | 'error',
  progress: number,        // 0.0 to 1.0
  message: string,         // Human-readable status message
  result_path?: string,    // Available when complete
  error?: string,          // Error message if failed
  current_frame?: number,  // For video processing
  total_frames?: number,   // For video processing
}
```

### Status Events
```typescript
{
  status: 'connected' | 'disconnected'
}
```

### Error Events
```typescript
{
  error: string
}
```

## Direct WebSocket Management

For more control, you can manage WebSocket connections directly:

```typescript
import {
  connectJobWebSocket,
  disconnectJobWebSocket,
  subscribeToJobUpdates,
  subscribeToJobStatus,
  subscribeToJobErrors,
} from '@/lib/preprocessor';

// Connect
await connectJobWebSocket(jobId);

// Subscribe to events
const unsubUpdate = subscribeToJobUpdates(jobId, (data) => {
  console.log('Update:', data);
});

const unsubStatus = subscribeToJobStatus(jobId, (data) => {
  console.log('Status:', data);
});

const unsubError = subscribeToJobErrors(jobId, (data) => {
  console.error('Error:', data);
});

// Later, unsubscribe and disconnect
unsubUpdate();
unsubStatus();
unsubError();
await disconnectJobWebSocket(jobId);
```

## React Hook Example

Here's a custom hook for managing preprocessor jobs:

```typescript
import { useState, useEffect, useCallback } from 'react';
import { PreprocessorJob } from '@/lib/preprocessor';

export function usePreprocessorJob(jobId: string | null) {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [result, setResult] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const job = new PreprocessorJob(jobId);

    job.onUpdate((data) => {
      if (data.progress !== undefined) {
        setProgress(data.progress);
      }
      if (data.message) {
        setStatus(data.message);
      }
      if (data.status === 'complete' && data.result_path) {
        setResult(data.result_path);
      }
      if (data.status === 'error') {
        setError(data.error || 'Unknown error');
      }
    });

    job.onError((data) => {
      setError(data.error);
    });

    job.connect().catch((err) => {
      setError(err.message);
    });

    return () => {
      job.disconnect();
    };
  }, [jobId]);

  return { progress, status, result, error };
}

// Usage in component:
function MyComponent() {
  const [jobId, setJobId] = useState<string | null>(null);
  const { progress, status, result, error } = usePreprocessorJob(jobId);

  return (
    <div>
      <p>Status: {status}</p>
      <p>Progress: {Math.round(progress * 100)}%</p>
      {result && <p>Result: {result}</p>}
      {error && <p>Error: {error}</p>}
    </div>
  );
}
```

## Error Handling

Always check the `success` field before accessing `data`:

```typescript
const result = await runPreprocessor({ ... });
if (result.success && result.data) {
  // Safe to use result.data
  const jobId = result.data.job_id;
} else {
  // Handle error
  console.error('Failed:', result.error);
}
```

## Notes

- WebSocket connections are automatically managed by the main process
- Multiple WebSocket connections can be active simultaneously for different jobs
- The backend uses Ray for distributed processing, allowing parallel job execution
- Job results are cached on the backend and can be retrieved even after disconnecting
- The `download_if_needed` parameter in `runPreprocessor` will automatically download models if not present
- Frame range parameters (`start_frame`, `end_frame`) only apply to video inputs

## Troubleshooting

### WebSocket Connection Fails
- Ensure the backend server is running
- Check that the backend URL is correctly configured
- Verify firewall settings allow WebSocket connections

### Job Never Completes
- Check job status using `getPreprocessorStatus(jobId)`
- Look for errors in the backend logs
- Verify the input file exists and is readable

### Download Fails
- Check available disk space
- Verify internet connection for model downloads
- Check backend logs for detailed error messages

## Example Component

See `apex/packages/renderer/src/components/preprocessor/PreprocessorExample.tsx` for a complete example of using these functions in a React component.

