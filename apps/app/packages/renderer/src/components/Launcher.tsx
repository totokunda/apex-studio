import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { launchMainWindow } from "@app/preload";

const Launcher: React.FC = () => {
  const [isLaunching, setIsLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onLaunch = async () => {
    setIsLaunching(true);
    setError(null);
    const res = await launchMainWindow();
    if (!res.ok) {
      setError(res.error || "Failed to launch");
      setIsLaunching(false);
    } else {
     setIsLaunching(false);
    }
    // On success, main process will close this window shortly.
  };

  return (
    <main className="w-full h-screen flex flex-col bg-black text-center font-poppins">
      <div className="relative flex-1 overflow-hidden">
        <div className="absolute inset-0 bg-linear-to-br from-slate-950 via-black to-slate-900" />
        <div className="absolute inset-0 backdrop-blur-md bg-black/50" />

        <div className="relative z-10 h-full w-full flex flex-col items-center justify-center gap-6 px-6">
          <div className="text-xs uppercase tracking-[0.35em] text-slate-400">
            Launcher
          </div>

          <div className="text-3xl font-semibold tracking-tight text-slate-50">
            Apex Studio
          </div>

          <p className="max-w-md text-xs text-slate-400 leading-relaxed">
            Click launch to open the main application.
          </p>

          <div className="flex flex-col items-center gap-3">
            <Button
              size="lg"
              className="bg-brand-accent text-white hover:bg-brand-accent-shade disabled:opacity-70"
              disabled={isLaunching}
              onClick={onLaunch}
            >
              {isLaunching ? "Launching..." : "Launch"}
            </Button>
            {error ? (
              <div className="text-xs text-red-300 max-w-md">{error}</div>
            ) : null}
          </div>
        </div>
      </div>
    </main>
  );
};

export default Launcher;


