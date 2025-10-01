import React, { useState } from 'react'
import { LuChevronDown, LuChevronUp, LuCheck} from "react-icons/lu";

import {
    DropdownMenu,
    DropdownMenuContent,
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

const TopBar:React.FC<TopBarProps> = () => {
    const layout = useLayoutConfigStore((s) => s.layout);
    const setLayout = useLayoutConfigStore((s) => s.setLayout);
    const [layoutOpen, setLayoutOpen] = useState(false);
    const aspectRatio = useViewportStore((s) => s.aspectRatio);
    const setAspectRatio = useViewportStore((s) => s.setAspectRatio);
    const [sizeOpen, setSizeOpen] = useState(false);
    const layoutLabel = layout === 'default' ? 'Default' : layout === 'media' ? 'Media' : 'Properties';

  return (
    <div className="w-full relative h-8 mt-2 px-6 flex items-center justify-end space-x-2">
      <DropdownMenu open={sizeOpen} onOpenChange={setSizeOpen}    >
                <DropdownMenuTrigger  className='text-brand-light/90 dark w-24 h-[34px] flex items-center space-x-1 px-2 relative font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-md py-[7px] transition-all duration-300 cursor-pointer'>
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
                <DropdownMenuTrigger className='text-brand-light/90 dark w-32 h-[34px] relative flex items-center space-x-2 px-2 font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand/90 rounded-md py-[7px] transition-all duration-300 cursor-pointer'>
                  <LayoutIcon type={layout as any} />
                  <span className='text-[11px]'>{layoutLabel}</span>
                  <div className='absolute right-2'>
                    {layoutOpen ? <LuChevronUp className='w-4 h-4' /> : <LuChevronDown className='w-4 h-4' />}
                  </div>
                </DropdownMenuTrigger>
                <DropdownMenuContent align='end' className='dark w-48 font-poppins'>
                  <DropdownMenuItem className='dark text-[12px] flex items-center gap-x-2' onClick={() => { setLayout('default'); setLayoutOpen(false); }}>
                    <LayoutIcon type='default' />
                    <span>Default</span>
                    {layout === 'default' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem className='dark text-[12px] flex items-center gap-x-2' onClick={() => { setLayout('media'); setLayoutOpen(false); }}>
                    <LayoutIcon type='media' />
                    <span>Media</span>
                    {layout === 'media' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem className='dark text-[12px] flex items-center gap-x-2' onClick={() => { setLayout('properties'); setLayoutOpen(false); }}>
                    <LayoutIcon type='properties' />
                    <span>Properties</span>
                    {layout === 'properties' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
                  </DropdownMenuItem>
                </DropdownMenuContent>
           </DropdownMenu>

            <button className='text-brand-light space-x-1.5 w-22 flex items-center justify-between px-3.5 font-medium h-[34px] hover:text-brand-light bg-brand-accent border border-brand-accent-two-shade hover:bg-brand-accent-two-shade rounded-md py-1.5 transition-all duration-300 cursor-pointer'>
            <span className='text-[11px]'>Export</span><LuChevronDown className='w-4 h-4' />
            </button>

      </div>
  )
}

export default TopBar