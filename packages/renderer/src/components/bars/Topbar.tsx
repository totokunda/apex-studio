import React, { useState } from 'react'
import { LuChevronDown, LuChevronUp, LuCheck} from "react-icons/lu";
import { SiShortcut } from "react-icons/si";

import {
    DropdownMenu,
    DropdownMenuContent,
  DropdownMenuLabel,
    DropdownMenuItem,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
  } from "@/components/ui/dropdown-menu"
import { useLayoutConfigStore } from "@/lib/layout-config";
import { useViewportStore } from '@/lib/viewport';
import { PiResize } from 'react-icons/pi';
import { LayoutIcon } from './LayoutIcon';
interface TopBarProps {

}

const Keycap: React.FC<{ label: string }> = ({ label }) => (
  <span className='inline-flex items-center justify-center px-1.5 h-5 min-w-[20px] shadow-sm rounded-[4px] bg-brand-light/[0.075] border border-brand-light/20 text-[10.5px]  text-brand-light/80'>
    {label}
  </span>
);

const ShortcutRow: React.FC<{ action: string; keys: string[] }> = ({ action, keys }) => (
  <div className='flex items-center justify-between'>
    <span className='text-[11px] text-brand-light/90 font-medium'>{action}</span>
    <div className='flex items-center gap-x-1.5'>
      {keys.map((k, i) => (
        <React.Fragment key={`${action}-${k}-${i}`}>
          <Keycap label={k} />
          {i < keys.length - 1 && <span className='text-brand-light/40 text-[10px]'>+</span>}
        </React.Fragment>
      ))}
    </div>
  </div>
);

const TopBar:React.FC<TopBarProps> = () => {
    const layout = useLayoutConfigStore((s) => s.layout);
    const setLayout = useLayoutConfigStore((s) => s.setLayout);
    const [layoutOpen, setLayoutOpen] = useState(false);
    const aspectRatio = useViewportStore((s) => s.aspectRatio);
    const setAspectRatio = useViewportStore((s) => s.setAspectRatio);
    const [sizeOpen, setSizeOpen] = useState(false);
    const [shortcutsOpen, setShortcutsOpen] = useState(false);
    const layoutLabel = layout === 'default' ? 'Default' : layout === 'media' ? 'Media' : 'Properties';

  return (
    <div className="w-full relative h-8 mt-2 px-6 flex items-center justify-end space-x-2">
      
           <DropdownMenu open={shortcutsOpen} onOpenChange={setShortcutsOpen}>
                <DropdownMenuTrigger className='text-brand-light/90 dark w-32 h-[34px] relative flex items-center space-x-2 px-2 font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-[6px] py-[7px] transition-all duration-300 cursor-pointer'>
                  <span className='text-[11px] inline-flex items-center space-x-1 w-full'><span className='text-brand-light/50 font-light'>⌘</span> <span>Shortcuts</span> 
                    </span>
                    {shortcutsOpen ? <LuChevronUp className='w-3.5 h-3.5' /> : <LuChevronDown className='w-3.5 h-3.5' />}
                  
                </DropdownMenuTrigger>
                <DropdownMenuContent align='end' className='dark w-[400px] font-poppins p-0 overflow-hidden bg-brand'>
                  <div className='bg-brand border-b border-brand-light/10 px-3 py-2'>
                    <span className='text-[11px] uppercase tracking-wide text-brand-light/70'>Keyboard Shortcuts</span>
                  </div>
                  <div className='divide-y divide-brand-light/10 '>
                  <div className='p-3'>
                      <div className='grid grid-cols-2 gap-x-4 gap-y-1.5'>
                        <ShortcutRow action='Zoom to Fit' keys={['⌘', '0']} />
                        <ShortcutRow action='Zoom to 50%' keys={['⌘', '5']} />
                        <ShortcutRow action='Zoom to 100%' keys={['⌘', '1']} />
                        <ShortcutRow action='Zoom to 200%' keys={['⌘', '2']} />
                        <ShortcutRow action='Group' keys={['⌘', 'G']} />
                        <ShortcutRow action='Ungroup' keys={['⌘', '⇧', 'G']} />
                      </div>
                    </div>
                    <div className='p-3'>
                      <DropdownMenuLabel className='dark text-[11px] font-semibold text-brand-light p-0'>Tools</DropdownMenuLabel>
                      <div className='mt-2 grid grid-cols-2 gap-x-4 gap-y-1'>
                        <ShortcutRow action='Pointer' keys={['V']} />
                        <ShortcutRow action='Hand' keys={['H']} />
                        <ShortcutRow action='Draw' keys={['D']} />
                        <ShortcutRow action='Text' keys={['T']} />
                        <ShortcutRow action='Mask' keys={['M']} />
                        <ShortcutRow action='Shape' keys={['S']} />
                      </div>
                    </div>
                    
                    <div className='p-3'>
                      <DropdownMenuLabel className='dark text-[11px] font-semibold text-brand-light p-0'>Shape Mode</DropdownMenuLabel>
                      <div className='mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5'>
                        <ShortcutRow action='Rectangle' keys={['1']} />
                        <ShortcutRow action='Ellipse' keys={['2']} />
                        <ShortcutRow action='Polygon' keys={['3']} />
                        <ShortcutRow action='Line' keys={['4']} />
                        <ShortcutRow action='Star' keys={['5']} />
                      </div>
                    </div>
                    <div className='p-3'>
                      <DropdownMenuLabel className='dark text-[11px] font-semibold text-brand-light p-0'>Mask Mode</DropdownMenuLabel>
                      <div className='mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5'>
                        <ShortcutRow action='Lasso' keys={['1']} />
                        <ShortcutRow action='Shape' keys={['2']} />
                        <ShortcutRow action='Draw' keys={['3']} />
                        <ShortcutRow action='Touch' keys={['4']} />
                      </div>
                    </div>
                    <div className='p-3'>
                        <DropdownMenuLabel className='dark text-[11px] font-semibold text-brand-light p-0'>Draw Mode</DropdownMenuLabel>
                      <div className='mt-2 grid grid-cols-2 gap-x-4 gap-y-1.5'>
                        <ShortcutRow action='Brush' keys={['1']} />
                        <ShortcutRow action='Highlighter' keys={['2']} />
                        <ShortcutRow action='Eraser' keys={['3']} />
                      </div>
                    </div>
                  </div>
                </DropdownMenuContent>
           </DropdownMenu>
           <DropdownMenu open={sizeOpen} onOpenChange={setSizeOpen}    >
                <DropdownMenuTrigger  className='text-brand-light/90 dark w-24 h-[34px] flex items-center space-x-1  px-2 relative font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-[6px] py-[7px] transition-all duration-300 cursor-pointer'>
                <PiResize className='w-4 h-4' /> <span className='text-xs'>Size</span>
                <div className='absolute right-2'>
                {sizeOpen ? <LuChevronUp className='w-3.5 h-3.5' /> : <LuChevronDown className='w-3.5 h-3.5' />}   
                </div>
                </DropdownMenuTrigger>
                <DropdownMenuContent align='end' className='dark w-48 flex flex-col bg-brand-background font-poppins'>
                  {[
                    { id: '16:9', name: 'Wide', w: 16, h: 9 },
                    { id: '9:16', name: 'Vertical', w: 9, h: 16 },
                    { id: '1:1', name: 'Square', w: 1, h: 1 },
                    { id: '4:3', name: 'Classic', w: 4, h: 3 },
                    { id: '4:5', name: 'Social', w: 4, h: 5 },
                    { id: '21:9', name: 'Cinema', w: 21, h: 9 },
                    { id: '2:3', name: 'Portrait', w: 2, h: 3 },
                  ].map((opt, index) => (
                    <>
                    <DropdownMenuItem
                      key={opt.id}
                      textValue={opt.name}
                      className='dark text-[12px] font-medium flex flex-row items-center cursor-pointer gap-x-3 w-full'
                      onClick={() => { setAspectRatio({ width: opt.w, height: opt.h, id: opt.id }); setSizeOpen(false); }}
                    >
                      <div
                        className='w-[24px] border-[1.5px] border-brand-light rounded-xs'
                        style={{ height: `${(24 * (opt.h / opt.w)).toFixed(1)}px` }}
                      />
                      <div className='flex flex-col items-start gap-y-0.5'>
                        <span>{opt.name}</span>
                        <span className='text-brand-light/50 text-xs font-light'>{opt.id}</span>
                      </div>
                      {aspectRatio.id === opt.id && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
                    </DropdownMenuItem>
                     {index !== 6 && <DropdownMenuSeparator />}
                     </>
                  ))}
                 
                </DropdownMenuContent>
           </DropdownMenu> 
           <DropdownMenu open={layoutOpen} onOpenChange={setLayoutOpen}>
                <DropdownMenuTrigger className='text-brand-light/90 dark w-32 h-[34px] relative flex items-center space-x-2 px-2 font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand/90 rounded-[6px] py-[7px] transition-all duration-300 cursor-pointer'>
                  <LayoutIcon type={layout as any} />
                  <span className='text-[11px]'>{layoutLabel}</span>
                  <div className='absolute right-2'>
                    {layoutOpen ? <LuChevronUp className='w-4 h-4' /> : <LuChevronDown className='w-4 h-4' />}
                  </div>
                </DropdownMenuTrigger>
                <DropdownMenuContent align='end' className='dark w-48 font-poppins bg-brand-background'>
                  <DropdownMenuItem className='dark text-[11px] font-medium flex items-center gap-x-2' onClick={() => { setLayout('default'); setLayoutOpen(false); }}>
                    <LayoutIcon type='default' />
                    <span>Default</span>
                    {layout === 'default' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem className='dark text-[11px] font-medium flex items-center gap-x-2' onClick={() => { setLayout('media'); setLayoutOpen(false); }}>
                    <LayoutIcon type='media' />
                    <span>Media</span>
                    {layout === 'media' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem className='dark text-[11px] font-medium flex items-center gap-x-2' onClick={() => { setLayout('properties'); setLayoutOpen(false); }}>
                    <LayoutIcon type='properties' />
                    <span>Properties</span>
                    {layout === 'properties' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
                  </DropdownMenuItem>
                </DropdownMenuContent>
           </DropdownMenu>

            <button className='text-brand-light space-x-1.5 w-28 flex items-center justify-between px-3.5 font-medium h-[34px] hover:text-brand-light bg-brand-accent border border-brand-accent-two-shade hover:bg-brand-accent-two-shade rounded-[6px] py-1.5 transition-all duration-300 cursor-pointer'>
            <span className='text-[11px]'>Export</span><LuChevronDown className='w-3.5 h-3.5' />
            </button>

      </div>
  )
}

export default TopBar