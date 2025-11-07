import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import { useContextMenuStore } from '@/lib/context-menu';
import { useClipStore } from '@/lib/clip';
import { useControlsStore } from '@/lib/control';
import { cn } from '@/lib/utils';

const Key: React.FC<{ text: string }> = ({ text }) => (
  <span className=' text-[10px] text-brand-light/60'>{text}</span>
);

const GlobalContextMenu: React.FC = () => {
  const { open, position, items, groups, closeMenu, setPosition, target } = useContextMenuStore();
  const containerRef = useRef<HTMLDivElement | null>(null);
  const clipsStore = useClipStore();

  useEffect(() => {
    if (!open) return;
    const onAny = (e: Event) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) closeMenu();
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') closeMenu(); };
    document.addEventListener('pointerdown', onAny, true);
    document.addEventListener('mousedown', onAny, true);
    document.addEventListener('click', onAny, true);
    document.addEventListener('contextmenu', onAny, true);
    window.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onAny, true);
      document.removeEventListener('mousedown', onAny, true);
      document.removeEventListener('click', onAny, true);
      document.removeEventListener('contextmenu', onAny, true);
      window.removeEventListener('keydown', onKey);
    };
  }, [open, closeMenu]);

  // Clamp to viewport bounds after mount
  useEffect(() => {
    if (!open) return;
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    let nx = position.x, ny = position.y;
    const BOTTOM_MARGIN = 10; // ensure some space from bottom edge
    if (rect.right > window.innerWidth) nx -= (rect.right - window.innerWidth);
    if (rect.bottom > window.innerHeight - BOTTOM_MARGIN) ny -= (rect.bottom - (window.innerHeight - BOTTOM_MARGIN));
    if (rect.left < 0) nx += -rect.left;
    if (rect.top < 0) ny += -rect.top;
    if (nx !== position.x || ny !== position.y) setPosition({ x: nx, y: ny });
  }, [open, position.x, position.y, setPosition]);

  if (!open) return null;

  const onSelect = (action: string) => {
    if (target?.type === 'clip') {
      const ids = target.clipIds;
      if (action === 'copy') clipsStore.copyClips(ids);
      else if (action === 'cut') { clipsStore.cutClips(ids); useControlsStore.getState().clearSelection(); }
      else if (action === 'paste') clipsStore.pasteClips(useControlsStore.getState().focusFrame);
      else if (action === 'delete') { ids.forEach(id => clipsStore.removeClip(id)); useControlsStore.getState().clearSelection(); }
      else if (action === 'split') clipsStore.splitClip(useControlsStore.getState().focusFrame, target.primaryClipId);
      else if (action === 'separateAudio' && target.isVideo) clipsStore.separateClip(target.primaryClipId);
      else if (action === 'export') { try { console.info('Export placeholder', ids); } catch {} }
      else if (action === 'group') {  clipsStore.groupClips(ids);}
      else if (action === 'ungroup') { clipsStore.ungroupClips(target.primaryClipId); }
      else if (action === 'convertToMedia') { clipsStore.convertToMedia(target.primaryClipId); }
    } else if (target?.type === 'timeline') {
      if (action === 'paste') {
        const frame = useControlsStore.getState().focusFrame;
        clipsStore.pasteClips(frame, target.timelineId);
        // Attempt to fix any overlaps by resolving again (pasteClips already resolves overlaps globally)
        // If needed, we could snap pasted clips to nearest valid gaps on target timeline here.
      }
    }
    closeMenu();
  };

  const content = groups && groups.length > 0 ? (
    <div className='p-1'>
      {groups.map((group, gi) => (
        group.items.length > 0 && <div key={group.id}>
          {group.label && <div className='px-2.5 py-1 text-[10px] uppercase tracking-wide text-brand-light/50'>{group.label}</div>}
          {group.items.map(item => (
            <button key={item.id} disabled={item.disabled} onClick={() => onSelect(item.action)} className={cn('w-full px-2.5 py-1.5 text-left text-[11.5px] flex items-center rounded justify-between hover:bg-brand-light/10 text-brand-light', item.disabled && 'opacity-50 !cursor-default hover:bg-brand')}>
              <span>{item.label}</span>
              {item.shortcut && <Key text={item.shortcut} />}
            </button>
          ))}
          {gi < groups.length - 1 && <div className='my-1 h-px bg-brand-light/10 -mx-1' />}
        </div>
      ))}
    </div>
  ) : (
    <div className='py-1'>
      {items.map(item => (
        <button key={item.id} disabled={item.disabled} onClick={() => onSelect(item.action)} className='w-full px-2.5 py-1.5 text-left text-[11px] flex items-center justify-between hover:bg-brand-light/10 disabled:opacity-50 text-brand-light'>
          <span>{item.label}</span>
          {item.shortcut && <Key text={item.shortcut} />}
        </button>
      ))}
    </div>
  );

  return createPortal(
    <div ref={containerRef} style={{ position: 'fixed', left: position.x, top: position.y, zIndex: 10000 }} className='w-52 font-poppins select-none rounded-md border border-brand-light/10 bg-brand-background-light shadow-lg'>
      {content}
    </div>,
    document.body
  );
};

export default GlobalContextMenu;


