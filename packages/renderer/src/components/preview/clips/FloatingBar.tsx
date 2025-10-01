import React, { useEffect, useMemo, useState } from 'react'
import { LuChevronDown, LuChevronUp, LuMousePointer2, LuHand, LuCheck, LuPen, LuSquareSquare, LuSquare, LuCircle, LuTriangle, LuMinus, LuStar, LuType} from "react-icons/lu";
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

// Removed individual MousePointerButton and HandButton in favor of a dropdown selector

const MaskButton = ({ active, onClick }: { active: boolean; onClick: () => void }) => {
  return (
    <div onClick={onClick} className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer ${active ? 'text-brand-light bg-brand-accent-two-shade' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'}`}>
      <LuSquareSquare className="w-5 h-5" />
    </div>
  )
}

const PenButton = ({ active, onClick }: { active: boolean; onClick: () => void }) => {
  return (
    <div onClick={onClick} className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer ${active ? 'text-brand-light bg-brand-accent-two-shade' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'}`}>
      <LuPen className="w-5 h-5" />
    </div>
  )
}

const TextButton = ({ active, onClick }: { active: boolean; onClick: () => void }) => {
  return (
    <div onClick={onClick} className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer ${active ? 'text-brand-light bg-brand-accent-two-shade' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'}`}>
      <LuType className="w-5 h-5" />
    </div>
  )
}

// Removed individual ShapeButton in favor of a dropdown selector for shapes

const FloatingBar:React.FC<FloatingBarProps> = () => {
    const scale = useViewportStore((s) => s.scale);
    const minScale = useViewportStore((s) => s.minScale);
    const maxScale = useViewportStore((s) => s.maxScale);
    const tool = useViewportStore((s) => s.tool);
    const shape = useViewportStore((s) => s.shape);
    const setTool = useViewportStore((s) => s.setTool);
    const setShape = useViewportStore((s) => s.setShape);
    const setScalePercent = useViewportStore((s) => s.setScalePercent);
    const centerContentAt = useViewportStore((s) => s.centerContentAt);
    
    const [zoomLevel, setZoomLevel] = useState<number>(Math.round(scale * 100));
    const [zoomOpen, setZoomOpen] = useState(false);
    
    const [shapeOpen, setShapeOpen] = useState(false);
    const [toolOpen, setToolOpen] = useState(false);
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
                {zoomOpen ? <LuChevronUp className='w-3.5 h-3.5' /> : <LuChevronDown className='w-3.5 h-3.5' />}   
                </DropdownMenuTrigger>
                <DropdownMenuContent className='dark w-60 bg-brand-background font-poppins'>
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
           
           
           </div>
           <div className="flex items-center gap-x-1">
           <PenButton active={tool === 'draw'} onClick={() => setTool('draw')} />
           <TextButton active={tool === 'text'} onClick={() => setTool('text')} />
          <MaskButton active={tool === 'mask'} onClick={() => setTool('mask')} />
          <div onClick={() => setTool('shape')} className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer ${
            tool === 'shape' ? 'text-brand-light bg-brand-accent-two-shade' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'
          }`}>
            {shape === 'rectangle' && <LuSquare className='w-5 h-5' />}
            {shape === 'ellipse' && <LuCircle className='w-5 h-5' />}
            {shape === 'polygon' && <LuTriangle className='w-5 h-5' />}
            {shape === 'line' && <LuMinus className='w-5 h-5' />}
            {shape === 'star' && <LuStar className='w-5 h-5' />}
          </div>
          <DropdownMenu open={shapeOpen} onOpenChange={setShapeOpen}>
            <DropdownMenuTrigger className={`rounded h-8 px-0.5 -ml-0.5 transition-all duration-300 cursor-pointer ${
              shapeOpen ? 'text-brand-light bg-brand-light/10' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'
            }`}>
              {shapeOpen ? (
                <LuChevronUp className="w-3.5 h-3.5" />
              ) : (
                <LuChevronDown className="w-3.5 h-3.5" />
              )}
            </DropdownMenuTrigger>
            <DropdownMenuContent className='dark w-44 bg-brand-background font-poppins'>
               <DropdownMenuItem
                 key='shape-rectangle'
                 textValue='Rectangle'
                 className='dark text-[12px] flex items-center gap-x-2'
                 onClick={() => { setShape('rectangle'); setTool('shape'); setShapeOpen(false); }}
               >
                 <LuSquare className='w-4 h-4' />
                 <span>Rectangle</span>
                 {shape === 'rectangle' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
               </DropdownMenuItem>
               <DropdownMenuSeparator />
               <DropdownMenuItem
                 key='shape-ellipse'
                 textValue='Ellipse'
                 className='dark text-[12px] flex items-center gap-x-2'
                 onClick={() => { setShape('ellipse'); setTool('shape'); setShapeOpen(false); }}
               >
                 <LuCircle className='w-4 h-4' />
                 <span>Ellipse</span>
                 {shape === 'ellipse' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
               </DropdownMenuItem>
               <DropdownMenuSeparator />
               <DropdownMenuItem
                 key='shape-polygon'
                 textValue='Polygon'
                 className='dark text-[12px] flex items-center gap-x-2'
                 onClick={() => { setShape('polygon'); setTool('shape'); setShapeOpen(false); }}
               >
                 <LuTriangle className='w-4 h-4' />
                 <span>Polygon</span>
                 {shape === 'polygon' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
               </DropdownMenuItem>
               <DropdownMenuSeparator />
               <DropdownMenuItem
                 key='shape-star'
                 textValue='Star'
                 className='dark text-[12px] flex items-center gap-x-2'
                 onClick={() => { setShape('star'); setTool('shape'); setShapeOpen(false); }}
               >
                 <LuStar className='w-4 h-4' />
                 <span>Star</span>
                 {shape === 'star' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
               </DropdownMenuItem>
             </DropdownMenuContent>
           </DropdownMenu>
            
          <div onClick={() => setTool(tool === 'hand' ? 'hand' : 'pointer')} className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer ${
            (tool === 'pointer' || tool === 'hand')
              ? 'text-brand-light bg-brand-accent-two-shade'
              : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'
          }`}>
            {tool === 'hand' ? (
              <LuHand className="w-5 h-5" />
            ) : (
              <LuMousePointer2 className="w-5 h-5" />
            )}
          </div>
          <DropdownMenu open={toolOpen} onOpenChange={setToolOpen}>
            <DropdownMenuTrigger className={`rounded py-2 px-0.5 -ml-0.5 h-8 transition-all duration-300 cursor-pointer ${
              toolOpen ? 'text-brand-light bg-brand-light/10' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'
            }`}>
              {toolOpen ? (
                <LuChevronUp className="w-3.5 h-3.5" />
              ) : (
                <LuChevronDown className="w-3.5 h-3.5" />
              )}
            </DropdownMenuTrigger>
             <DropdownMenuContent className='dark w-36 bg-brand-background font-poppins '>
               <DropdownMenuItem
                 key='tool-pointer'
                 textValue='Pointer'
                 className='dark text-[12px] flex items-center gap-x-2'
                 onClick={() => { setTool('pointer'); setToolOpen(false); }}
               >
                 <LuMousePointer2 className='w-4 h-4' />
                 <span>Pointer</span>
                 {tool === 'pointer' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
               </DropdownMenuItem>
               <DropdownMenuSeparator />
               <DropdownMenuItem
                 key='tool-hand'
                 textValue='Hand'
                 className='dark text-[12px] flex items-center gap-x-2'
                 onClick={() => { setTool('hand'); setToolOpen(false); }}
               >
                 <LuHand className='w-4 h-4' />
                 <span>Hand</span>
                 {tool === 'hand' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
               </DropdownMenuItem>
             </DropdownMenuContent>
           </DropdownMenu>
            
            </div>
          </div>

        </div>
      </div>
  )
}

export default FloatingBar