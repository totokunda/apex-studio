import React, { useState, useEffect } from "react";
import { saveImageToPath, resolveAssetPath } from "@app/preload";

const HF_BASE_URL = "https://huggingface.co/datasets/totoku/apex-studio-images/resolve/main/";

interface FallbackAssetProps {
  src?: string;
  type?: "img" | "video";
  fallbackBase?: string;
  alt?: string;
  className?: string;
  poster?: string;
  [key: string]: any;
}

export const FallbackAsset: React.FC<FallbackAssetProps> = ({ 
  src, 
  type = "img", 
  fallbackBase = HF_BASE_URL, 
  alt,
  className,
  poster,
  ...props 
}) => {
  const [currentSrc, setCurrentSrc] = useState(src);
  const [currentPoster, setCurrentPoster] = useState(poster);
  const [hasFailed, setHasFailed] = useState(false);

  useEffect(() => {
    setCurrentSrc(src);
    setHasFailed(false);
  }, [src]);

  useEffect(() => {
    setCurrentPoster(poster);
  }, [poster]);

  const downloadAsset = async (assetPath: string) => {
    if (!assetPath) return null;
    
    // Construct HF URL
    let relativePath = assetPath;
    if (relativePath.startsWith("/")) {
      relativePath = relativePath.slice(1);
    }
    
    // If it's already an absolute URL (and not localhost), we don't need fallback
    if (/^https?:\/\//i.test(assetPath) && !assetPath.includes("localhost") && !assetPath.includes("127.0.0.1")) {
        return assetPath;
    }

    const hfUrl = `${fallbackBase}${relativePath}`;

    // Attempt to download to local intended path so it's available next time
    try {
      const absPath = await resolveAssetPath(assetPath);
      if (absPath) {
        const resp = await fetch(hfUrl);
        if (resp.ok) {
          const buffer = await resp.arrayBuffer();
          await saveImageToPath(buffer, absPath);
          console.log(`[FallbackAsset] Successfully downloaded missing asset to ${absPath}`);
        }
      }
    } catch (err) {
      console.warn(`[FallbackAsset] Failed to download fallback asset for ${assetPath}`, err);
    }
    
    return hfUrl;
  };

  const handleError = async () => {
    if (hasFailed || !src) return;
    setHasFailed(true);

    const hfUrl = await downloadAsset(src);
    if (hfUrl) {
      setCurrentSrc(hfUrl);
    }

    // Also try to download the poster if it's a video
    if (type === "video" && poster) {
      const hfPosterUrl = await downloadAsset(poster);
      if (hfPosterUrl) {
        setCurrentPoster(hfPosterUrl);
      }
    }
  };

  if (type === "video") {
    return (
      <video 
        {...props} 
        src={currentSrc} 
        poster={currentPoster}
        onError={handleError} 
        className={className}
      />
    );
  }

  return (
    <img 
      {...props} 
      src={currentSrc} 
      onError={handleError} 
      alt={alt}
      className={className}
    />
  );
};

export default FallbackAsset;
