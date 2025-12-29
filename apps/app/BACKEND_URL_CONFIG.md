# Backend URL Configuration

## Overview

The backend API URL is now configurable, allowing users to change the port or even point to a remote backend server.

## Storage

The backend URL is persisted in a JSON file located at:

```
{userData}/apex-settings.json
```

Where `{userData}` is Electron's userData directory, typically:

- **macOS**: `~/Library/Application Support/apex/`
- **Windows**: `%APPDATA%/apex/`
- **Linux**: `~/.config/apex/`

## Default Value

If no custom URL is configured, the default is:

```
http://127.0.0.1:8765
```

## Settings File Format

```json
{
  "backendUrl": "http://127.0.0.1:8765"
}
```

## API Usage

### From Renderer (Recommended)

```typescript
import { getBackendApiUrl, setBackendApiUrl } from "@/lib/config";

// Get current URL
const result = await getBackendApiUrl();
if (result.success) {
  console.log("Current URL:", result.data.url);
}

// Set new URL
const updateResult = await setBackendApiUrl("http://127.0.0.1:9000");
if (updateResult.success) {
  console.log("URL updated to:", updateResult.data.url);
}
```

### From Preload

```typescript
import { getBackendUrl, setBackendUrl } from "@app/preload";

const result = await getBackendUrl();
const updateResult = await setBackendUrl("http://192.168.1.100:8765");
```

## Behavior

1. **On App Start**: The settings file is loaded and the backend URL is read
2. **On URL Change**: The new URL is immediately saved to the settings file
3. **Validation**: URLs are validated using the standard URL constructor
4. **Immediate Effect**: Changes take effect immediately for all subsequent API calls

## Use Cases

- **Different Port**: Run backend on a custom port
- **Remote Server**: Point to a backend running on another machine
- **Development**: Switch between local and remote backends easily
- **Multiple Instances**: Run multiple backend instances on different ports

## Manual Configuration

You can also manually edit the `apex-settings.json` file to configure the backend URL. Make sure the app is closed before editing, or your changes may be overwritten.

## Troubleshooting

If API calls fail after changing the backend URL:

1. Verify the backend is running at the specified URL
2. Check the URL format is correct (must include protocol: `http://` or `https://`)
3. Ensure there's no firewall blocking the connection
4. Try resetting to the default: `http://127.0.0.1:8765`
5. Check the settings file for syntax errors

## Security Considerations

- The backend URL is stored in plain text in the settings file
- Ensure you trust the backend server you're connecting to
- For remote connections, consider using HTTPS if available
- The app does not validate SSL certificates by default (Node.js fetch behavior)
