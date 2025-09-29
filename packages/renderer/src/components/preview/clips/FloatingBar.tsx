import React, { useEffect, useMemo, useState } from 'react'
import { LuChevronDown, LuChevronUp, LuMousePointer2, LuHand, LuCheck, LuPen, LuSquareSquare, LuStar} from "react-icons/lu";
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


interface FloatingBarProps {
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

const MaskButton = ({ active, onClick }: { active: boolean; onClick: () => void }) => {
  return (
    <div onClick={onClick} className={`rounded-md p-1.5 transition-all duration-300 cursor-pointer ${active ? 'text-brand-light bg-brand-light/10' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'}`}>
      <LuSquareSquare className="w-5 h-5" />
    </div>
  )
}

const PenButton = ({ active, onClick }: { active: boolean; onClick: () => void }) => {
  return (
    <div onClick={onClick} className={`rounded-md p-1.5 transition-all duration-300 cursor-pointer ${active ? 'text-brand-light bg-brand-light/10' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'}`}>
      <LuPen className="w-5 h-5" />
    </div>
  )
}

const ShapeButton = ({ active, onClick }: { active: boolean; onClick: () => void }) => {
  return (
    <div onClick={onClick} className={`rounded-md p-1.5 transition-all duration-300 cursor-pointer ${active ? 'text-brand-light bg-brand-light/10' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'}`}>
      <span className='text-xs'><LuStar className="w-5 h-5" /></span>
    </div>
  )
}

const FloatingBar:React.FC<FloatingBarProps> = () => {
    const scale = useViewportStore((s) => s.scale);
    const minScale = useViewportStore((s) => s.minScale);
    const maxScale = useViewportStore((s) => s.maxScale);
    const tool = useViewportStore((s) => s.tool);
    const setTool = useViewportStore((s) => s.setTool);
    const setScalePercent = useViewportStore((s) => s.setScalePercent);
    const centerContentAt = useViewportStore((s) => s.centerContentAt);
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
    <div className="w-64 absolute top-7 left-1/2 -translate-x-1/2 rounded-lg px-6 z-50 ">
        <div className="w-full h-full flex items-center justify-between">

          <div className="flex flex-row-reverse justify-center items-center gap-x-1.5 absolute top-0 left-1/2 -translate-x-1/2 p-2 bg-brand border border-brand-light/5 rounded-lg shadow-lg">
            
            <div className="flex items-center gap-x-1">
            <DropdownMenu open={zoomOpen} onOpenChange={setZoomOpen}    >
                <DropdownMenuTrigger  className='text-brand-light/90 dark w-18  flex items-center font-medium justify-between px-2 text-xs border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-md py-[7px] transition-all duration-300 cursor-pointer'>
                {zoomLevel}%
                {zoomOpen ? <LuChevronUp className='w-4 h-4' /> : <LuChevronDown className='w-4 h-4' />}   
                </DropdownMenuTrigger>
                <DropdownMenuContent className='dark w-60 bg-brand-background'>
                  <DropdownMenuLabel className='flex flex-col justify-center py-2 pb-0 px-1.5'>
                    <span className='text-brand-light text-xs'>Size</span>
                    <div className='flex flex-row items-center gap-x-2 w-full'>
                    <Slider className='w-full dark' value={[zoomLevel]} max={sliderMax} min={sliderMin} step={1} onValueChange={(value) => {setZoomLevel(value[0]); setScalePercent(value[0]); }} />
                    <input  className='w-[42px] h-6 px-1 text-brand-light text-xs font-light items-center justify-center rounded-sm bg-brand-background' value={`${zoomLevel}%`} onChange={(e) => {
                      const raw = e.target.value.replace(/[^0-9]/g, '');
                      const num = Math.max(sliderMin, Math.min(sliderMax, Math.abs(parseInt(raw || '0'))));
                      if (!Number.isNaN(num)) setZoomLevel(num);
                    }} onBlur={() => setScalePercent(zoomLevel)} />
                    </div>
                  </DropdownMenuLabel>  
                  <DropdownMenuSeparator />
                  <DropdownMenuItem key='zoom-to-fit' textValue='Zoom to fit' className='dark text-[13px]' onClick={() => { centerContentAt(75); setZoomOpen(false); }}>Zoom to Fit</DropdownMenuItem>
                  <DropdownMenuItem key='zoom-to-50' textValue='Zoom to 50%' className='dark text-[13px]' onClick={() => { setScalePercent(50); setZoomOpen(false); }}>Zoom to 50%</DropdownMenuItem>
                  <DropdownMenuItem key='zoom-to-100' textValue='Zoom to 100%' className='dark text-[13px]' onClick={() => { setScalePercent(100); setZoomOpen(false); }}>Zoom to 100%</DropdownMenuItem>
                  <DropdownMenuItem key='zoom-to-200' textValue='Zoom to 200%' className='dark text-[13px]' onClick={() => { setScalePercent(200); setZoomOpen(false); }}>Zoom to 200%</DropdownMenuItem>
                </DropdownMenuContent>
           </DropdownMenu> 
           <DropdownMenu open={sizeOpen} onOpenChange={setSizeOpen}    >
                <DropdownMenuTrigger  className='text-brand-light/90 dark w-24  flex items-center space-x-1 px-2 relative font-medium border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-md py-[7px] transition-all duration-300 cursor-pointer'>
                <PiResize className='w-4 h-4' /> <span className='text-xs'>Size</span>
                <div className='absolute right-2'>
                {sizeOpen ? <LuChevronUp className='w-4 h-4' /> : <LuChevronDown className='w-4 h-4' />}   
                </div>
                </DropdownMenuTrigger>
                <DropdownMenuContent className='dark w-48 flex flex-col bg-brand-background'>
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
           <div className="flex items-center gap-x-1">
            <ShapeButton active={tool === 'shape'} onClick={() => setTool('shape')} />
            <PenButton active={tool === 'draw'} onClick={() => setTool('draw')} />
            <MaskButton active={tool === 'mask'} onClick={() => setTool('mask')} />
            <HandButton active={tool === 'hand'} onClick={() => setTool('hand')} />
            <MousePointerButton active={tool === 'pointer'} onClick={() => setTool('pointer')} />
            
            </div>
          </div>

        </div>
      </div>
  )
}

export default FloatingBar