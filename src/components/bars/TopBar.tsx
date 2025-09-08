import React, { useEffect, useMemo, useState } from 'react'
import { LuChevronDown, LuChevronUp, LuMousePointer2, LuHand, LuCheck} from "react-icons/lu";
import { PiResize } from "react-icons/pi";


import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
  } from "@/components/ui/dropdown-menu"
import { Slider } from "@/components/ui/slider"
import { useViewportStore } from "@/lib/viewport";
interface TopBarProps {
  title: string
}

const MousePointerButton = ({ active, onClick }: { active: boolean; onClick: () => void }) => {
  return (
    <div onClick={onClick} className={`rounded-md p-1.5 transition-all duration-300 cursor-pointer ${active ? 'text-brand-light bg-brand-light/10' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'}`}>
      <LuMousePointer2 className="w-5 h-5" />
    </div>
  )
}

const HandButton = ({ active, onClick }: { active: boolean; onClick: () => void }) => {
  return (
    <div onClick={onClick} className={`rounded-md p-1.5 transition-all duration-300 cursor-pointer ${active ? 'text-brand-light bg-brand-light/10' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'}`}>
      <LuHand className="w-5 h-5" />
    </div>
  )
}

const TopBar:React.FC<TopBarProps> = ({ title }) => {
    const scale = useViewportStore((s) => s.scale);
    const minScale = useViewportStore((s) => s.minScale);
    const maxScale = useViewportStore((s) => s.maxScale);
    const tool = useViewportStore((s) => s.tool);
    const setTool = useViewportStore((s) => s.setTool);
    const setScalePercentCentered = useViewportStore((s) => s.setScalePercentCentered);
    const centerContentAt100 = useViewportStore((s) => s.centerContentAt100);
    const aspectRatio = useViewportStore((s) => s.aspectRatio);
    const setAspectRatio = useViewportStore((s) => s.setAspectRatio);
    const [zoomLevel, setZoomLevel] = useState<number>(Math.round(scale * 100));
    const [zoomOpen, setZoomOpen] = useState(false);
    const [sizeOpen, setSizeOpen] = useState(false);
    useEffect(() => {
      setZoomLevel(Math.round(scale * 100));
    }, [scale])
    const sliderMin = useMemo(() => Math.round(minScale * 100), [minScale]);
    const sliderMax = useMemo(() => Math.round(maxScale * 100), [maxScale]);
  return (
    <div className="w-full h-12 px-6 bg-brand absolute top-0 left-0">
        <div className="w-full h-full flex items-center justify-between">
            <div>
          <h3 className="text-brand-light text-sm"></h3>
          </div>
          <div className="flex items-center gap-x-1.5 absolute left-1/2 -translate-x-1/2">
            <div className="flex items-center gap-x-1 mr-3">
            <HandButton active={tool === 'hand'} onClick={() => setTool('hand')} />
            <MousePointerButton active={tool === 'pointer'} onClick={() => setTool('pointer')} />
            </div>
            <DropdownMenu open={zoomOpen} onOpenChange={setZoomOpen}    >
                <DropdownMenuTrigger  className='text-brand-light/90 dark w-18  flex items-center font-medium justify-between px-2 text-xs border border-brand-light/10 hover:text-brand-light bg-brand-background/50 hover:bg-brand-light/10 rounded-md py-[7px] transition-all duration-300 cursor-pointer'>
                {zoomLevel}%
                {zoomOpen ? <LuChevronUp className='w-4 h-4' /> : <LuChevronDown className='w-4 h-4' />}   
                </DropdownMenuTrigger>
                <DropdownMenuContent className='dark w-60'>
                  <DropdownMenuLabel className='flex flex-col justify-center py-2 pb-0 px-1.5'>
                    <span className='text-brand-light text-xs'>Size</span>
                    <div className='flex flex-row items-center gap-x-2 w-full'>
                    <Slider className='w-full dark' value={[zoomLevel]} max={sliderMax} min={sliderMin} step={1} onValueChange={(value) => {setZoomLevel(value[0]); setScalePercentCentered(value[0]); }} />
                    <input  className='w-[42px] h-6 px-1 text-brand-light text-xs font-light items-center justify-center rounded-sm bg-brand-background/50' value={`${zoomLevel}%`} onChange={(e) => {
                      const raw = e.target.value.replace(/[^0-9]/g, '');
                      const num = Math.max(sliderMin, Math.min(sliderMax, Math.abs(parseInt(raw || '0'))));
                      if (!Number.isNaN(num)) setZoomLevel(num);
                    }} onBlur={() => setScalePercentCentered(zoomLevel)} />
                    </div>
                  </DropdownMenuLabel>  
                  <DropdownMenuSeparator />
                  <DropdownMenuItem key='zoom-to-fit' textValue='Zoom to fit' className='dark text-[13px]' onClick={() => { centerContentAt100(); setZoomOpen(false); }}>Zoom to Fit</DropdownMenuItem>
                  <DropdownMenuItem key='zoom-to-50' textValue='Zoom to 50%' className='dark text-[13px]' onClick={() => { setScalePercentCentered(50); setZoomOpen(false); }}>Zoom to 50%</DropdownMenuItem>
                  <DropdownMenuItem key='zoom-to-100' textValue='Zoom to 100%' className='dark text-[13px]' onClick={() => { setScalePercentCentered(100); setZoomOpen(false); }}>Zoom to 100%</DropdownMenuItem>
                  <DropdownMenuItem key='zoom-to-200' textValue='Zoom to 200%' className='dark text-[13px]' onClick={() => { setScalePercentCentered(200); setZoomOpen(false); }}>Zoom to 200%</DropdownMenuItem>
                </DropdownMenuContent>
           </DropdownMenu> 
           <DropdownMenu open={sizeOpen} onOpenChange={setSizeOpen}    >
                <DropdownMenuTrigger  className='text-brand-light/90 dark w-24  flex items-center space-x-1 px-2 font-medium border border-brand-light/10 hover:text-brand-light bg-brand-background/50 hover:bg-brand-light/10 rounded-md py-[7px] transition-all duration-300 cursor-pointer'>
                <PiResize className='w-4 h-4' /> <span className='text-xs'>Size</span>
                <div className='absolute right-2'>
                {sizeOpen ? <LuChevronUp className='w-4 h-4' /> : <LuChevronDown className='w-4 h-4' />}   
                </div>
                </DropdownMenuTrigger>
                <DropdownMenuContent className='dark w-48 flex flex-col '>
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
                      className='dark text-[13px] flex flex-row items-center cursor-pointer gap-x-3 w-full'
                      onClick={() => { setAspectRatio({ width: opt.w, height: opt.h, id: opt.id }); setSizeOpen(false); }}
                    >
                      <div
                        className='w-[24px] border-2 border-brand-light rounded'
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
          </div>
          <div>
            <button className='text-brand-light space-x-1.5 flex items-center justify-between px-3.5 font-medium  hover:text-brand-light bg-brand-accent hover:bg-brand-accent-hover rounded py-1.5 transition-all duration-300 cursor-pointer'>
            <span className='text-xs'>Export</span><LuChevronDown className='w-3 h-3' />
            </button>
          </div>
        </div>
      </div>
  )
}

export default TopBar