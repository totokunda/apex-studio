import React, { useState, useEffect } from "react";
import {
  getBackendApiUrl,
  setBackendApiUrl,
  getApexHomeDir,
  setApexHomeDir,
  getApexTorchDevice,
  setApexTorchDevice,
  getApexCachePath,
  setApexCachePath,
} from "@/lib/config";

/**
 * Example component demonstrating usage of the config API functions.
 * This can be integrated into a settings page or used as a reference.
 */
export function ConfigExample() {
  const [backendUrl, setBackendUrlState] = useState<string>("");
  const [homeDir, setHomeDirState] = useState<string>("");
  const [torchDevice, setTorchDeviceState] = useState<string>("");
  const [cachePath, setCachePathState] = useState<string>("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadCurrentConfig();
  }, []);

  const loadCurrentConfig = async () => {
    setLoading(true);
    try {
      const [backendUrlRes, homeDirRes, torchDeviceRes, cachePathRes] =
        await Promise.all([
          getBackendApiUrl(),
          getApexHomeDir(),
          getApexTorchDevice(),
          getApexCachePath(),
        ]);

      if (backendUrlRes.success && backendUrlRes.data) {
        setBackendUrlState(backendUrlRes.data.url);
      }
      if (homeDirRes.success && homeDirRes.data) {
        setHomeDirState(homeDirRes.data.home_dir);
      }
      if (torchDeviceRes.success && torchDeviceRes.data) {
        setTorchDeviceState(torchDeviceRes.data.device);
      }
      if (cachePathRes.success && cachePathRes.data) {
        setCachePathState(cachePathRes.data.cache_path);
      }
    } catch (error) {
      console.error("Failed to load config:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleSetBackendUrl = async () => {
    const newUrl = prompt("Enter backend API URL:", backendUrl);
    if (!newUrl) return;

    const result = await setBackendApiUrl(newUrl);
    if (result.success && result.data) {
      setBackendUrlState(result.data.url);
      alert("Backend URL updated successfully!");
    } else {
      alert(`Error: ${result.error}`);
    }
  };

  const handleSetHomeDir = async () => {
    const newPath = prompt("Enter new home directory:", homeDir);
    if (!newPath) return;

    const result = await setApexHomeDir(newPath);
    if (result.success && result.data) {
      setHomeDirState(result.data.home_dir);
      alert("Home directory updated successfully!");
    } else {
      alert(`Error: ${result.error}`);
    }
  };

  const handleSetTorchDevice = async () => {
    const newDevice = prompt(
      "Enter torch device (cpu, cuda, mps, cuda:0):",
      torchDevice,
    );
    if (!newDevice) return;

    const result = await setApexTorchDevice(newDevice);
    if (result.success && result.data) {
      setTorchDeviceState(result.data.device);
      alert("Torch device updated successfully!");
    } else {
      alert(`Error: ${result.error}`);
    }
  };

  const handleSetCachePath = async () => {
    const newPath = prompt("Enter new cache path:", cachePath);
    if (!newPath) return;

    const result = await setApexCachePath(newPath);
    if (result.success && result.data) {
      setCachePathState(result.data.cache_path);
      alert("Cache path updated successfully!");
    } else {
      alert(`Error: ${result.error}`);
    }
  };

  if (loading) {
    return <div>Loading configuration...</div>;
  }

  return (
    <div style={{ padding: "20px" }}>
      <h2>Apex Configuration</h2>

      <div style={{ marginBottom: "20px" }}>
        <h3>Backend API URL</h3>
        <p>{backendUrl}</p>
        <button onClick={handleSetBackendUrl}>Change Backend URL</button>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <h3>Home Directory</h3>
        <p>{homeDir}</p>
        <button onClick={handleSetHomeDir}>Change Home Directory</button>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <h3>Torch Device</h3>
        <p>{torchDevice}</p>
        <button onClick={handleSetTorchDevice}>Change Torch Device</button>
      </div>

      <div style={{ marginBottom: "20px" }}>
        <h3>Cache Path</h3>
        <p>{cachePath}</p>
        <button onClick={handleSetCachePath}>Change Cache Path</button>
      </div>

      <button onClick={loadCurrentConfig}>Refresh Configuration</button>
    </div>
  );
}
