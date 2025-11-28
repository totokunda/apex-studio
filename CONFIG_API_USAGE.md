# Config API Usage Guide

This document explains how to use the Apex config API functions in the renderer.

## Overview

The config API allows the renderer to interact with backend configuration settings for:

- Backend API URL configuration
- Home directory management
- Torch device configuration (CPU, CUDA, MPS)
- Cache path settings

## Architecture

The implementation follows Electron's IPC pattern:

1. **Backend API** (`apex-engine/src/api/config.py`): FastAPI endpoints (default port 8765)
2. **Main Process** (`apex/packages/main/src/modules/ApexApi.ts`): IPC handlers that make HTTP requests
3. **Preload Script** (`apex/packages/preload/src/index.ts`): Exposes IPC functions via contextBridge
4. **Renderer Helpers** (`apex/packages/renderer/src/lib/config/`): Typed wrapper functions

## Available Functions

### Get Backend API URL

```typescript
import { getBackendApiUrl } from "@/lib/config";

const result = await getBackendApiUrl();
if (result.success && result.data) {
  console.log("Backend URL:", result.data.url);
} else {
  console.error("Error:", result.error);
}
```

### Set Backend API URL

```typescript
import { setBackendApiUrl } from "@/lib/config";

const result = await setBackendApiUrl("http://127.0.0.1:9000");
if (result.success && result.data) {
  console.log("Backend URL updated to:", result.data.url);
} else {
  console.error("Error:", result.error);
}
```

### Get Home Directory

```typescript
import { getApexHomeDir } from "@/lib/config";

const result = await getApexHomeDir();
if (result.success && result.data) {
  console.log("Home dir:", result.data.home_dir);
} else {
  console.error("Error:", result.error);
}
```

### Set Home Directory

```typescript
import { setApexHomeDir } from "@/lib/config";

const result = await setApexHomeDir("/path/to/new/home");
if (result.success && result.data) {
  console.log("Updated to:", result.data.home_dir);
} else {
  console.error("Error:", result.error);
}
```

### Get Torch Device

```typescript
import { getApexTorchDevice } from "@/lib/config";

const result = await getApexTorchDevice();
if (result.success && result.data) {
  console.log("Current device:", result.data.device);
}
```

### Set Torch Device

```typescript
import { setApexTorchDevice } from "@/lib/config";

// Valid values: 'cpu', 'cuda', 'mps', 'cuda:0', 'cuda:1', etc.
const result = await setApexTorchDevice("cuda");
if (result.success && result.data) {
  console.log("Device set to:", result.data.device);
} else {
  console.error("Error:", result.error);
}
```

### Get Cache Path

```typescript
import { getApexCachePath } from "@/lib/config";

const result = await getApexCachePath();
if (result.success && result.data) {
  console.log("Cache path:", result.data.cache_path);
}
```

### Set Cache Path

```typescript
import { setApexCachePath } from "@/lib/config";

const result = await setApexCachePath("/path/to/cache");
if (result.success && result.data) {
  console.log("Cache path updated to:", result.data.cache_path);
}
```

## Response Format

All functions return a `ConfigResponse<T>` object:

```typescript
interface ConfigResponse<T> {
  success: boolean;
  data?: T; // Present when success is true
  error?: string; // Present when success is false
}
```

## Example Component

See `apex/packages/renderer/src/components/config/ConfigExample.tsx` for a complete example of using these functions in a React component.

## Direct Import from Preload

You can also import directly from the preload layer:

```typescript
import { getBackendUrl, setBackendUrl, getHomeDir, setHomeDir, getTorchDevice, setTorchDevice, getCachePath, setCachePath } from "@app/preload";
```

However, using the wrapper functions from `@/lib/config` is recommended as they provide better type hints and documentation.

## Error Handling

Always check the `success` field before accessing `data`:

```typescript
const result = await getApexHomeDir();
if (result.success && result.data) {
  // Safe to use result.data
} else {
  // Handle error using result.error
  console.error("Failed:", result.error);
}
```

## Notes

- The backend API URL is configurable and defaults to `http://127.0.0.1:8765`
- The backend URL is persisted in `apex-settings.json` in the app's userData directory
- Changes to the backend URL take effect immediately for subsequent API calls
- Some settings (like home directory) may require an application restart to take full effect
- Invalid device names or URLs will return an error response with details
