import React from 'react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { LuChevronDown, LuArrowUpDown, LuLoader } from "react-icons/lu";
import { LuCloudUpload } from "react-icons/lu";
import { toast } from 'sonner';
import { TbFileUpload, TbFolderUp} from "react-icons/tb";
import { ScrollArea } from "@/components/ui/scroll-area"
import Item, { MediaItem } from '@/components/media/Item'

import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
  } from "@/components/ui/dropdown-menu"
import { cn } from '@/lib/utils';
import { BsFilter } from "react-icons/bs";
import { DropdownMenuCheckboxItem } from "@/components/ui/dropdown-menu";
import { listConvertedMedia, importMediaPaths, ensureUniqueConvertedName, renameMediaPair, deleteMediaPair, getLowercaseExtension, pickMediaPaths } from '@app/preload';
import { getMediaInfo } from '@/lib/media/utils';
import { VIDEO_EXTS, IMAGE_EXTS, AUDIO_EXTS } from '@/lib/settings';
 

function isSupported(p: string) {
    try {
        const e = getLowercaseExtension(p);
        return new Set(VIDEO_EXTS).has(e) || new Set(IMAGE_EXTS).has(e) || new Set(AUDIO_EXTS).has(e);
    } catch (e) {
        return false;
    }
}

async function pickFilesViaInput(directory: boolean): Promise<string[] | undefined> {

  const filters = [
    { name: 'All Supported Files', extensions: VIDEO_EXTS.concat(IMAGE_EXTS).concat(AUDIO_EXTS) },
    { name: 'Video Files', extensions: VIDEO_EXTS },
    { name: 'Image Files', extensions: IMAGE_EXTS },
    { name: 'Audio Files', extensions: AUDIO_EXTS },
  ];

  const title = directory ? 'Choose a folder to import' : 'Choose file(s) to import';
  const res = await pickMediaPaths({ directory, filters, title });
  return res?.filter((p) => isSupported(p)) ?? [];
}

interface MediaSidebarProps {
    onClose: () => void;
}


// (unused with Electron file picker)

// ===== Filtering & Sorting helpers (outside component) =====
type MediaType = 'video' | 'image' | 'audio';
type SortKey = 'name' | 'duration' | 'date';
type SortOrder = 'asc' | 'desc';

const filterMediaItems = (items: MediaItem[], selectedTypes: Set<MediaType>): MediaItem[] => {
  if (!selectedTypes || selectedTypes.size === 0) return items;
  return items.filter((it) => selectedTypes.has(it.type as MediaType));
};

const nameComparator = (a: MediaItem, b: MediaItem) => a.name.toLowerCase().localeCompare(b.name.toLowerCase());
const dateComparator = (a: MediaItem, b: MediaItem) => (a.dateAddedMs ?? 0) - (b.dateAddedMs ?? 0);
const durationComparator = (
  a: MediaItem,
  b: MediaItem,
  durations: Record<string, number | undefined>
) => {
  const da = durations[a.assetUrl];
  const db = durations[b.assetUrl];
  const na = Number.isFinite(da as number) ? (da as number) : Number.POSITIVE_INFINITY;
  const nb = Number.isFinite(db as number) ? (db as number) : Number.POSITIVE_INFINITY;
  return na - nb;
};

const sortMediaItems = (
  items: MediaItem[],
  sortKey: SortKey,
  sortOrder: SortOrder,
  durations: Record<string, number | undefined>
): MediaItem[] => {
  const sorted = [...items];
  let cmp: (a: MediaItem, b: MediaItem) => number;
  switch (sortKey) {
    case 'duration':
      cmp = (a, b) => durationComparator(a, b, durations);
      break;
    case 'date':
      cmp = dateComparator;
      break;
    case 'name':
    default:
      cmp = nameComparator;
      break;
  }
  sorted.sort(cmp);
  if (sortOrder === 'desc') sorted.reverse();
  return sorted;
};



// interface DeleteAlertDialogProps {
//   onDelete: () => void;
//   open: boolean;
//   onOpenChange: (open: boolean) => void;
// }


const MediaSidebar: React.FC<MediaSidebarProps> = ({ onClose }) => {
    const [items, setItems] = useState<MediaItem[]>([]);
    const [loading, setLoading] = useState(false);
    const panelRef = useRef<HTMLDivElement | null>(null);
    const listContainerRef = useRef<HTMLDivElement | null>(null);
    const uploadBarRef = useRef<HTMLDivElement | null>(null);
    const [menuWidth, setMenuWidth] = useState<number>(0);
    const [panelHeight, setPanelHeight] = useState<number>(0);
    const [deleteAlertOpen, setDeleteAlertOpen] = useState(false);
    const [renamingItem, setRenamingItem] = useState<string | null>(null);
    const [renameValue, setRenameValue] = useState<string>("");
    const renameInputRef = useRef<HTMLInputElement | null>(null);
    const renameContainerRef = useRef<HTMLDivElement | null>(null);
    const [deleteItem, setDeleteItem] = useState<string | null>(null);
    const [selectedTypes, setSelectedTypes] = useState<Set<MediaType>>(new Set());
    const [sortKey, setSortKey] = useState<SortKey>('name');
    const [sortOrder, setSortOrder] = useState<SortOrder>('asc');
    const [filterOpen, setFilterOpen] = useState(false);
    const [sortOpen, setSortOpen] = useState(false);
    const [durationCache, setDurationCache] = useState<Record<string, number | undefined>>({});
    const [isUploading, setIsUploading] = useState(false);

    useEffect(() => {
      const el = uploadBarRef.current;
      if (!el) return;
      const update = () => setMenuWidth(el.clientWidth);
      update();
      const ro = new ResizeObserver(update);
      ro.observe(el);
      window.addEventListener('resize', update);
      return () => {
        ro.disconnect();
        window.removeEventListener('resize', update);
      };
    }, []);

    // Track panel height and update on resize to size the ScrollArea dynamically (panelHeight - 172px)
    useEffect(() => {
      const el = panelRef.current;
      if (!el) return;
      const update = () => setPanelHeight(el.clientHeight);
      update();
      const ro = new ResizeObserver(update);
      ro.observe(el);
      window.addEventListener('resize', update);
      return () => {
        ro.disconnect();
        window.removeEventListener('resize', update);
      };
    }, []);
    
    const loadMediaList = useCallback(async () => {
      try {
        setLoading(true);
        const list = await listConvertedMedia();
        const results: MediaItem[] = [];
        for (const it of list) {
          const info = await getMediaInfo(it.assetUrl);
          results.push({ name: it.name, type: it.type, absPath: it.absPath, assetUrl: it.assetUrl, dateAddedMs: it.dateAddedMs, mediaInfo: info });
        }
        results.sort((a, b) => a.name.toLowerCase().localeCompare(b.name.toLowerCase()));
        setItems(results);
      } catch (e) {
        console.error(e);
      } finally {
        setLoading(false);
      }
    }, []);

    useEffect(() => { loadMediaList(); }, [loadMediaList]);
    
    const handleUpload = async (directory: boolean = false) => {
        try {
          setIsUploading(true);
          const paths = await pickFilesViaInput(directory);
          if (!paths || paths.length === 0) return;
          const loadingId = toast.loading(`Importing ${paths.length} item(s)…`, { position: "bottom-right" });
          await importMediaPaths(paths, '480p');
          toast.dismiss(loadingId);
          await loadMediaList();
        } catch (error) {
          console.error(error);
          toast.error("Failed to upload file", {
            position: "bottom-right",
            duration: 3000,
            style: {
              width: 'fit-content'
            }
          });
        }
        finally {
          setIsUploading(false);
        }
      };
    
      // ensures "a (1).mp4", "a (2).mp4", etc., on converted side
      const ensureUniqueName = async (desired: string) => {
        return ensureUniqueConvertedName(desired);
      };

      const sanitizeBaseName = (name: string) => {
        const trimmed = name.trim();
        // Replace invalid characters for cross-platform compatibility
        const replaced = trimmed.replace(/[\\/:*?"<>|]/g, "-");
        // Collapse whitespace
        const collapsed = replaced.replace(/\s+/g, " ");
        // Remove trailing dots/spaces which are invalid on Windows
        const noTrail = collapsed.replace(/[ .]+$/g, "");
        // Limit length
        const limited = noTrail.slice(0, 128);
        return limited || "Untitled";
      };

      

      const startRename = (item: MediaItem) => {
        const dot = item.name.lastIndexOf('.');
        const base = dot >= 0 ? item.name.slice(0, dot) : item.name;
        setRenamingItem(item.name);
        setRenameValue(base);
        // focus happens on render via autoFocus; small delay to ensure selection
        setTimeout(() => {
          renameInputRef.current?.select();
        }, 0);
      };

      const commitRename = async (originalName: string) => {
        const dot = originalName.lastIndexOf('.');
        const ext = dot >= 0 ? originalName.slice(dot) : "";
        // const oldRel = await join(BASE_MEDIA_DIR, '24', originalName);
        const sanitizedBase = sanitizeBaseName(renameValue.replace(new RegExp(`${ext.replace('.', '\\.')}$`), ""));
        const oldBase = dot >= 0 ? originalName.slice(0, dot) : originalName;
        if (sanitizedBase === oldBase) {
          setRenamingItem(null);
          return;
        }
        const unique = await ensureUniqueName(`${sanitizedBase}${ext}`);
        // const newRel = await join(BASE_MEDIA_DIR, '24', unique);
        try {
          await renameMediaPair(originalName, unique);
          toast.success(`Renamed to ${unique}`, { position: "bottom-right", duration: 3000, style:{ width: 'fit-content' } });
          setRenamingItem(null);
          setRenameValue("");
          await loadMediaList();
        } catch (e) {
          console.error(e);
          toast.error("Failed to rename", { position: "bottom-right", duration: 3000, style:{ width: 'fit-content' } });
          setRenamingItem(null);
        }
      };

      useEffect(() => {
        if (!renamingItem) return;
        const onDocPointerDown = (ev: PointerEvent) => {
          const container = renameContainerRef.current;
          if (!container) return;
          if (!container.contains(ev.target as Node)) {
            commitRename(renamingItem);
          }
        };
        document.addEventListener('pointerdown', onDocPointerDown, true);
        return () => {
          document.removeEventListener('pointerdown', onDocPointerDown, true);
        };
      }, [renamingItem, commitRename]);

      const handleDelete = useCallback(async (name: string) => {
        try {
          await deleteMediaPair(name);
          toast.success(`Deleted ${name}`, {
            position: "bottom-right",
            duration: 3000,
            style: {
              width: 'fit-content'
            }
          });
          await loadMediaList();
        } catch (e) {
          console.error(e);
          toast.error('Failed to delete', {
            position: "bottom-right",
            duration: 3000,
            style: {
              width: 'fit-content'
            }
          });
        }
      }, [loadMediaList]);

      // removed old inline thumbnail implementations in favor of memoized MediaThumb above
      // Prefetch durations lazily when sorting by duration
      useEffect(() => {
        if (sortKey !== 'duration') return;
        let cancelled = false;
        (async () => {
          const targets = items.filter((it) => (it.type === 'video' || it.type === 'audio') && durationCache[it.assetUrl] == null);
          for (const it of targets) {
            try {
              const info = await getMediaInfo(it.assetUrl);
              if (cancelled) return;
              setDurationCache((prev) => ({ ...prev, [it.assetUrl]: info?.duration ?? undefined }));
            } catch {}
          }
        })();
        return () => { cancelled = true };
      }, [sortKey, items]);

      const displayItems = useMemo(() => {
        const f = filterMediaItems(items, selectedTypes);
        return sortMediaItems(f, sortKey, sortOrder, durationCache);
      }, [items, selectedTypes, sortKey, sortOrder, durationCache]);
    
  return (
    <div ref={panelRef} className="h-full w-full  duration-200 ease-out ">
        <div className="flex flex-col gap-y-3 px-5 pb-5 pt-3">

        <DropdownMenu>
        <div ref={uploadBarRef} className="bg-brand-accent-shade shadow flex flex-row  items-center   text-brand-lighter w-full text-[12.5px] font-medium rounded transition-all duration-200 cursor-pointer">
            <span  onClick={() => handleUpload(false)} className='px-4 py-2 flex flex-row gap-x-2.5 rounded-l items-center justify-center hover:bg-brand-accent-two-shade border-r border-brand-light/30 w-full'>
            {isUploading ? (
              <LuLoader className="w-[18px] h-[18px] animate-spin" />
            ) : (
              <LuCloudUpload className="w-[18px] h-[18px] cursor-pointer stroke-2" />
            )}
            <span className="flex flex-row gap-x-2.5 items-center justify-center">
                {isUploading ? 'Uploading…' : 'Upload Media'}
            </span>
            </span>
            <DropdownMenuTrigger  className='px-1 py-2 flex flex-row gap-x-2.5 items-center justify-center w-10 hover:bg-brand-accent-two-shade rounded-r-md'>
                <LuChevronDown className="w-4 h-4 cursor-pointer stroke-2" />
            </DropdownMenuTrigger>
            <DropdownMenuContent align='end' sideOffset={2} style={{ width: menuWidth, maxWidth: menuWidth }} className='dark mt-1 flex flex-col text-brand-light bg-brand-background'>
                <DropdownMenuItem onClick={() => handleUpload(false)} className='w-full cursor-pointer py-2'>
                    <TbFileUpload className="w-[18px] h-[18px] cursor-pointer stroke-2" />
                    <span className="flex flex-row gap-x-2.5 items-center justify-center text-[12.5px]">
                        Upload File
                    </span>
                </DropdownMenuItem>
                <DropdownMenuItem    onClick={() => handleUpload(true)} className='w-full cursor-pointer py-2'>
                    <TbFolderUp className="w-[18px] h-[18px] cursor-pointer stroke-2" />
                    <span className="flex flex-row gap-x-2.5 items-center justify-center text-[12.5px]">
                        Upload Directory
                    </span>
                </DropdownMenuItem>
            </DropdownMenuContent>
        </div>
        </DropdownMenu>

      </div>
      <div className="border-t border-brand-light/5"></div>
      {/* Filter & Sort Toolbar */}
      <div className="px-5 py-2">
        <div className="flex flex-row items-center gap-x-1.5">
          { /* compute selected state for styling */ }
          { /* eslint-disable-next-line */ }
          {/* toolbar */}
          <DropdownMenu open={filterOpen} onOpenChange={setFilterOpen}>
            <DropdownMenuTrigger asChild>
              <button
                className={cn(
                  "px-2.5 py-1.5 rounded-md text-brand-light/90 text-[12px] flex flex-row items-center gap-x-2 transition-colors",
                  "hover:bg-brand-light/10",
                  ((filterOpen) || (selectedTypes.size > 0)) && "bg-brand-light/10",
                )}
                title="Filter"
              >
                <BsFilter className="w-[16px] h-[16px]" />
                <span>Filter</span>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align='start' className='dark w-44 flex flex-col text-brand-light bg-brand-background font-poppins text-[12px]'>
              <DropdownMenuCheckboxItem className='text-[12px]'
                checked={selectedTypes.size === 0}
                onCheckedChange={(checked) => {
                  if (checked) setSelectedTypes(new Set());
                }}
              >All</DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem className='text-[12px]'
                checked={selectedTypes.has('video')}
                onCheckedChange={(checked) => {
                  setSelectedTypes((prev) => {
                    const next = new Set(prev);
                    if (checked) next.add('video'); else next.delete('video');
                    return next;
                  });
                }}
              >Videos</DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem className='text-[12px]'
                checked={selectedTypes.has('image')}
                onCheckedChange={(checked) => {
                  setSelectedTypes((prev) => {
                    const next = new Set(prev);
                    if (checked) next.add('image'); else next.delete('image');
                    return next;
                  });
                }}
              >Images</DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem className='text-[12px]'
                checked={selectedTypes.has('audio')}
                onCheckedChange={(checked) => {
                  setSelectedTypes((prev) => {
                    const next = new Set(prev);
                    if (checked) next.add('audio'); else next.delete('audio');
                    return next;
                  });
                }}
              >Audio</DropdownMenuCheckboxItem>
            </DropdownMenuContent>
          </DropdownMenu>
          <DropdownMenu open={sortOpen} onOpenChange={setSortOpen}>
            <DropdownMenuTrigger asChild>
              <button
                className={cn(
                  "px-2.5 py-1.5 rounded-md text-brand-light/90 text-[12px] flex flex-row items-center gap-x-2 transition-colors",
                  "hover:bg-brand-light/10",
                  ((sortOpen)) && "bg-brand-light/10",
                )}
                title="Sort"
              >
                <LuArrowUpDown className="w-[16px] h-[16px]" />
                <span>Sort</span>
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align='start' className='dark w-48 flex flex-col text-brand-light bg-brand-background font-poppins'>
              {/* Sort key (mutually exclusive via checkboxes) */}
              <DropdownMenuCheckboxItem className='text-[12px]'
                checked={sortKey === 'name'}
                onCheckedChange={(checked) => { if (checked) setSortKey('name'); }}
              >Name</DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem className='text-[12px]'
                checked={sortKey === 'duration'}
                onCheckedChange={(checked) => { if (checked) setSortKey('duration'); }}
              >Duration</DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem className='text-[12px]'
                checked={sortKey === 'date'}
                onCheckedChange={(checked) => { if (checked) setSortKey('date'); }}
              >Date Added</DropdownMenuCheckboxItem>
              <DropdownMenuSeparator />
              {/* Sort order (mutually exclusive via checkboxes) */}
              <DropdownMenuCheckboxItem className='text-[12px]'
                checked={sortOrder === 'asc'}
                onCheckedChange={(checked) => { if (checked) setSortOrder('asc'); }}
              >Ascending</DropdownMenuCheckboxItem>
              <DropdownMenuCheckboxItem className='text-[12px]'
                checked={sortOrder === 'desc'}
                onCheckedChange={(checked) => { if (checked) setSortOrder('desc'); }}
              >Descending</DropdownMenuCheckboxItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div ref={listContainerRef} className=" overflow-y-auto">
        {loading && (
          <div className="text-brand-lighter/70 text-xs px-5 py-4 pt-1">Loading media…</div>
        )}
        {!loading && items.length === 0 && (
          <div className="text-brand-lighter/60 text-xs px-5 py-4 pt-1">No media yet. Upload files to get started.</div>
        )}
        {!loading && items.length > 0 && (
          <ScrollArea style={{ height: Math.max(0, panelHeight - 172) }} className="px-5 py-4 pt-1 dark"
          >
            <div className="grid gap-2 w-full" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(140px, 1fr))' }}>
            {displayItems.map((item) => (
              <Item key={item.name} 
              item={item} 
              renamingItem={renamingItem} 
              setRenamingItem={setRenamingItem} 
              renameContainerRef={renameContainerRef} 
              renameInputRef={renameInputRef} renameValue={renameValue}
              setRenameValue={setRenameValue} commitRename={commitRename} 
              setDeleteItem={setDeleteItem} setDeleteAlertOpen={setDeleteAlertOpen} 
              deleteAlertOpen={deleteAlertOpen} deleteItem={deleteItem} 
              handleDelete={handleDelete} startRename={startRename} />
            ))}
            </div>
          </ScrollArea>
        )}
      </div>
      
    </div>
  )
}

export default MediaSidebar