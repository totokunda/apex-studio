import React, { createContext, useContext, useState } from "react";
import VideoDecoderModule from "./module";
import { useMount } from "react-use";

const VideoDecoderContext = createContext<VideoDecoderModule | null>(null);

export const VideoDecoderProvider: React.FC<{
    children: React.ReactNode;
}> = ({ children }) => {
    const [module, setModule] = useState<VideoDecoderModule | null>(null);

    useMount(() => {
        const module = new VideoDecoderModule();
        setModule(module);
    });

    if (!module) {
        return null;
    }

    return React.createElement(
        VideoDecoderContext.Provider,
        { value: module },
        children
    );
};

export function useVideoDecoder(): VideoDecoderModule {
    const module = useContext(VideoDecoderContext);
    if (!module) {
        throw new Error("VideoDecoderModule not found");
    }
    return module;
}