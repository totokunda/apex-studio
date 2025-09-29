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
interface TopBarProps {

}

const TopBar:React.FC<TopBarProps> = () => {
    const layout = useLayoutConfigStore((s) => s.layout);
    const setLayout = useLayoutConfigStore((s) => s.setLayout);
    const [layoutOpen, setLayoutOpen] = useState(false);

  return (
    <div className="w-full relative h-8 mt-2 px-6 flex items-center justify-end space-x-2">

           <DropdownMenu open={layoutOpen} onOpenChange={setLayoutOpen}>
                <DropdownMenuTrigger className='text-brand-light/90 dark w-28 relative flex items-center space-x-1 px-2 font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand/90 rounded-md py-[7px] transition-all duration-300 cursor-pointer'>
                  <span className='text-[11px]'>Layout</span>
                  <div className='absolute right-2'>
                    {layoutOpen ? <LuChevronUp className='w-4 h-4' /> : <LuChevronDown className='w-4 h-4' />}
                  </div>
                </DropdownMenuTrigger>
                <DropdownMenuContent align='end' className='dark w-44 font-poppins'>
                  <DropdownMenuItem className='dark text-[11px] flex items-center gap-x-2' onClick={() => { setLayout('default'); setLayoutOpen(false); }}>
                    <span>Default</span>
                    {layout === 'default' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem className='dark text-[11px] flex items-center gap-x-2' onClick={() => { setLayout('media'); setLayoutOpen(false); }}>
                    <span>Media</span>
                    {layout === 'media' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem className='dark text-[11px] flex items-center gap-x-2' onClick={() => { setLayout('properties'); setLayoutOpen(false); }}>
                    <span>Properties</span>
                    {layout === 'properties' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
                  </DropdownMenuItem>
                </DropdownMenuContent>
           </DropdownMenu>

            <button className='text-brand-light space-x-1.5 flex items-center justify-between px-3.5 font-medium  hover:text-brand-light bg-brand-accent hover:bg-brand-accent-two-shade rounded py-1.5 transition-all duration-300 cursor-pointer'>
            <span className='text-[11px]'>Export</span><LuChevronDown className='w-3 h-3' />
            </button>

      </div>
  )
}

export default TopBar