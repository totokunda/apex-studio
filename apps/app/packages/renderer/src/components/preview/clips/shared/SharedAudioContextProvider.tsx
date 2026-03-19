import React, { createContext, useContext,  useMemo } from "react";

const SharedAudioContext = createContext<AudioContext | null>(null);

export const SharedAudioContextProvider: React.FC<{
  children: React.ReactNode;
}> = ({ children }) => {
  const audioContext = useMemo(() => {
    const AudioContext: typeof window.AudioContext =
      (window as any).AudioContext || (window as any).webkitAudioContext
    // Use low latency hint for minimal audio delay
    const ctx = new AudioContext({ latencyHint: "interactive"});
    
    return ctx;
  }, []);
  
  return <SharedAudioContext.Provider value={audioContext}>{children}</SharedAudioContext.Provider>;
};

export const useSharedAudioContext = () => {
  const context = useContext(SharedAudioContext);
  return {
    ctx: context || null,
  }
};