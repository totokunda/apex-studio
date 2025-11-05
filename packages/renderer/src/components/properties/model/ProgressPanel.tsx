import { useJobProgress, useEngineJobActions } from '../../../lib/engine/hooks'
import { useClipStore } from '@/lib/clip'
import { ModelClipProps } from '@/lib/types'
import { useEffect } from 'react'

interface ProgressPanelProps {
  clipId: string | null
}

const ProgressPanel: React.FC<ProgressPanelProps> = ({ clipId }) => {
  const job = useJobProgress(clipId)
  const clip = useClipStore((s) => s.getClipById(clipId ?? '')) as ModelClipProps | undefined;
  const { startTracking } = useEngineJobActions();

  // Ensure tracking stays active even if this panel unmounts
  useEffect(() => {
    if (!clipId) return;
    try { startTracking(clipId); } catch {}
  }, [clipId, startTracking]);

  const seenUpdateKeys = new Set<string>()
  const displayUpdates = ((job?.updates || []))
    .slice()
    .reverse()
    .filter((u) => {
      const pct = typeof u.progress === 'number' ? Math.round(u.progress) : undefined
      const msg = (u.message || job?.currentStep || 'Working...').toString().trim().replace(/\s+/g, ' ')
      const status = (u as any)?.status || ''
      const key = `${msg}|${pct ?? ''}|${status}`
      if (seenUpdateKeys.has(key)) return false
      seenUpdateKeys.add(key)
      return true
    })


  return (
    <div className="p-4">
      <div className="text-brand-light text-[11px] uppercase tracking-wide mb-3 text-start font-medium"> {clip?.manifest?.metadata?.name} Progress</div>
      {displayUpdates.length > 0 && <ul className="space-y-3">
        {displayUpdates.map((u, idx) => {
          const pct = typeof u.progress === 'number' ? Math.round(u.progress) : undefined
          const d = u.time ? new Date(u.time) : null
          const dateStr = d ? d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) : ''
          const timeStr = d ? d.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' }) : ''
          const key = `${job?.jobId || clipId}-u-${idx}`
          return (
            <li key={key} className="rounded-[6px] border border-brand-light/10 bg-brand p-2.5">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="text-[10.5px] font-medium text-brand-light truncate text-start">{(u.message || job?.currentStep || 'Working...').trim().replace(/\s+/g, ' ')}</div>
                </div>
                <div className="flex items-center gap-2">
                  {typeof pct === 'number' && (
                    <div className="text-[10.5px] font-medium text-brand-light">{pct}%</div>
                  )}
                </div>
              </div>
              <div className="mt-1 flex items-center justify-between text-[10px] text-brand-light/60">
                <span className="text-start">{dateStr}</span>
                <span className="text-end">{timeStr}</span>
              </div>
              {typeof pct === 'number' && (
                <div className="mt-2 h-1.5 w-full rounded bg-brand-light/10">
                  <div className="h-1.5 rounded bg-brand-accent-two-shade" style={{ width: `${Math.max(0, Math.min(100, pct))}%` }} />
                </div>
              )}
            </li>
          )
        })}
      </ul>}
      {(displayUpdates.length === 0) && (
        <div className="text-[10.5px] text-brand-light/60 p-2.5 text-start w-full bg-brand rounded-[6px] border border-brand-light/10">No updates found.</div>
      )}
    </div>
  )
}

export default ProgressPanel;