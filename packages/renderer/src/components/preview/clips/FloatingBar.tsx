import React, { useEffect, useMemo, useState } from 'react'
import { LuChevronDown, LuChevronUp, LuMousePointer2, LuHand, LuCheck, LuSquareSquare, LuSquare, LuCircle, LuTriangle, LuMinus, LuStar, LuType, LuHighlighter, LuEraser, LuPenTool} from "react-icons/lu";

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
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover';
import { useDrawingStore } from '@/lib/drawing';
import ColorInput from '@/components/properties/ColorInput';


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
    
    const drawingTool = useDrawingStore((s) => s.tool);
    const setDrawingTool = useDrawingStore((s) => s.setTool);
    const brushSize = useDrawingStore((s) => s.brushSize);
    const highlighterSize = useDrawingStore((s) => s.highlighterSize);
    const eraserSize = useDrawingStore((s) => s.eraserSize);
    const smoothing = useDrawingStore((s) => s.smoothing);
    const setBrushSize = useDrawingStore((s) => s.setBrushSize);
    const setHighlighterSize = useDrawingStore((s) => s.setHighlighterSize);
    const setEraserSize = useDrawingStore((s) => s.setEraserSize);
    const setSmoothing = useDrawingStore((s) => s.setSmoothing);
    const drawingColor = useDrawingStore((s) => s.color);
    const drawingOpacity = useDrawingStore((s) => s.opacity);
    const setDrawingColor = useDrawingStore((s) => s.setColor);
    const setDrawingOpacity = useDrawingStore((s) => s.setOpacity);
    
    const getCurrentSize = () => {
      switch (drawingTool) {
        case 'brush': return brushSize;
        case 'highlighter': return highlighterSize;
        case 'eraser': return eraserSize;
        default: return brushSize;
      }
    };
    
    const [zoomLevel, setZoomLevel] = useState<number>(Math.round(scale * 100));
    const [zoomOpen, setZoomOpen] = useState(false);
    
    const [shapeOpen, setShapeOpen] = useState(false);
    const [toolOpen, setToolOpen] = useState(false);
    const [drawOpen, setDrawOpen] = useState(false);
    const [tempSizeValue, setTempSizeValue] = useState<string>('3');
    
    useEffect(() => {
      setZoomLevel(Math.round(scale * 100));
    }, [scale])
    
    useEffect(() => {
      setTempSizeValue(getCurrentSize().toString());
    }, [drawingTool, brushSize, highlighterSize, eraserSize, getCurrentSize])
    const sliderMin = useMemo(() => Math.round(minScale * 100), [minScale]);
    const sliderMax = useMemo(() => Math.round(maxScale * 100), [maxScale]);
  return (
    <div className="w-64 absolute top-7 left-1/2 -translate-x-1/2 rounded-lg px-6 z-50 ">
        <div className="w-full h-full flex items-center justify-between">
          <div className="flex flex-row-reverse justify-center items-center gap-x-1.5 absolute top-0 left-1/2 -translate-x-1/2 p-2 bg-brand border border-brand-light/5 rounded-lg shadow-lg">
            <div className="flex items-center gap-x-1">
            <DropdownMenu open={zoomOpen} onOpenChange={setZoomOpen}>
                <DropdownMenuTrigger  className='text-brand-light/90 dark w-18  flex items-center font-medium justify-between px-2 text-xs border border-brand-light/10 hover:text-brand-light bg-brand hover:bg-brand-light/10 rounded-md py-[7px] transition-all duration-300 cursor-pointer'>
                {zoomLevel}%
                {zoomOpen ? <LuChevronUp className='w-3.5 h-3.5' /> : <LuChevronDown className='w-3.5 h-3.5' />}   
                </DropdownMenuTrigger>
                <DropdownMenuContent className='dark w-60 bg-brand-background font-poppins'>
                  <DropdownMenuLabel className='flex flex-col justify-center py-2 pb-0 px-1.5'>
                    <span className='text-brand-light text-xs font-medium'>Size</span>
                    <div className='flex flex-row items-center gap-x-2 w-full mb-1'>
                    <Slider className='w-full dark' value={[zoomLevel]} max={sliderMax} min={sliderMin} step={1} onValueChange={(value) => {setZoomLevel(value[0]); setScalePercent(value[0]); }} />
                    <input  className='w-[42px] h-7 border border-brand-light/10 px-1 bg-brand  text-brand-light focus:outline-brand-background text-center ring-brand-background  text-[11px] font-light items-center justify-center rounded-sm ' value={`${zoomLevel}%`} onChange={(e) => {
                      const raw = e.target.value.replace(/[^0-9]/g, '');
                      const num = Math.max(sliderMin, Math.min(sliderMax, Math.abs(parseInt(raw || '0'))));
                      if (!Number.isNaN(num)) setZoomLevel(num);
                    }} onBlur={() => setScalePercent(zoomLevel)} />
                    </div>
                  </DropdownMenuLabel>  
                  <DropdownMenuSeparator />
                  <DropdownMenuItem key='zoom-to-fit' textValue='Zoom to fit' className='dark text-[12px] font-medium' onClick={() => { centerContentAt(75); setZoomOpen(false); }}>Zoom to Fit</DropdownMenuItem>
                  <DropdownMenuItem key='zoom-to-50' textValue='Zoom to 50%' className='dark text-[12px] font-medium' onClick={() => { setScalePercent(50); setZoomOpen(false); }}>Zoom to 50%</DropdownMenuItem>
                  <DropdownMenuItem key='zoom-to-100' textValue='Zoom to 100%' className='dark text-[12px] font-medium' onClick={() => { setScalePercent(100); setZoomOpen(false); }}>Zoom to 100%</DropdownMenuItem>
                  <DropdownMenuItem key='zoom-to-200' textValue='Zoom to 200%' className='dark text-[12px] font-medium' onClick={() => { setScalePercent(200); setZoomOpen(false); }}>Zoom to 200%</DropdownMenuItem>
                </DropdownMenuContent>
           </DropdownMenu> 
           </div>
           <div className="flex items-center gap-x-1">
           <div onClick={() => setTool('draw')} className={`rounded-md h-8 w-8 p-1.5 transition-all duration-300 cursor-pointer ${tool === 'draw' ? 'text-brand-light bg-brand-accent-two-shade' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'}`}>
             {drawingTool === 'brush' && <LuPenTool className="w-5 h-5" />}
             {drawingTool === 'highlighter' && <LuHighlighter className="w-5 h-5" />}
             {drawingTool === 'eraser' && <LuEraser className="w-5 h-5" />}
            </div>
           <Popover open={drawOpen} onOpenChange={setDrawOpen}>
              <PopoverTrigger className={`rounded h-8 px-0.5 -ml-0.5 transition-all duration-300 cursor-pointer ${
                drawOpen ? 'text-brand-light bg-brand-light/10' : 'text-brand-light/90 hover:text-brand-light hover:bg-brand-light/10'
              }`}>
                {drawOpen ? (
                  <LuChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <LuChevronDown className="w-3.5 h-3.5" />
                )}
              </PopoverTrigger>
              <PopoverContent className='dark w-56 bg-brand-background font-poppins p-3'>
                <div className="flex flex-col gap-y-3">
                  {/* Drawing Tools */}
                  <div className="flex flex-col gap-y-0.5">
                    <label className="text-brand-light text-[11px] mb-1 font-medium">Tool</label>
                    <div className="flex flex-row border divide-x border-brand-light/10 rounded-md overflow-hidden bg-brand">
                      <div
                        onClick={() => { setTool('draw'); setDrawingTool('brush'); }}
                        className={`flex items-center gap-x-3 px-3 py-1.5 w-1/3 cursor-pointer justify-center transition-all duration-200 ${
                          drawingTool === 'brush' ? 'bg-brand-accent-two-shade text-brand-light' : 'text-brand-light/80 hover:bg-brand-light/5'
                        }`}
                      >
                        <LuPenTool className="w-4 h-4" />
                      </div>
                      <div
                        onClick={() => { setTool('draw'); setDrawingTool('highlighter'); }}
                        className={`flex items-center gap-x-3 px-3 py-1.5 w-1/3 justify-center cursor-pointer transition-all duration-200 ${
                          drawingTool === 'highlighter' ? 'bg-brand-accent-two-shade text-brand-light' : 'text-brand-light/80 hover:bg-brand-light/5'
                        }`}
                      >
                        <LuHighlighter className="w-4 h-4" />
                      </div>
                      <div
                        onClick={() => { setTool('draw'); setDrawingTool('eraser'); }}
                        className={`flex items-center gap-x-3 px-3 py-1.5 w-1/3 justify-center cursor-pointer transition-all duration-200 ${
                          drawingTool === 'eraser' ? 'bg-brand-accent-two-shade text-brand-light' : 'text-brand-light/80 hover:bg-brand-light/5'
                        }`}
                      >
                        <LuEraser className="w-4 h-4" />
                      </div>
                    </div>
                  </div>
                  {/* Size Slider */}
                  <div className="flex flex-col gap-y-0.5">
                    <label className="text-brand-light text-[12px] font-medium">Size</label>
                    <div className="flex items-center gap-x-2">
                      <Slider
                        className="w-full dark"
                        value={[getCurrentSize()]}
                        max={drawingTool === 'highlighter' ? 50 : drawingTool === 'eraser' ? 100 : 30}
                        min={1}
                        step={1}
                        onValueChange={(value) => {
                          if (drawingTool === 'brush') setBrushSize(value[0]);
                          else if (drawingTool === 'highlighter') setHighlighterSize(value[0]);
                          else if (drawingTool === 'eraser') setEraserSize(value[0]);
                        }}
                      />
                      <input
                        className="w-[40px] h-6 px-1 text-brand-light focus:outline-brand-background border border-brand-light/10 text-center bg-brand ring-brand-background text-[11px] font-normal items-center justify-center rounded"
                        value={tempSizeValue}
                        onChange={(e) => setTempSizeValue(e.target.value)}
                        onBlur={() => {
                          const raw = tempSizeValue.replace(/[^0-9]/g, '');
                          const maxVal = drawingTool === 'highlighter' ? 50 : drawingTool === 'eraser' ? 100 : 20;
                          const num = Math.max(1, Math.min(maxVal, Math.abs(parseInt(raw || '1'))));
                          if (!Number.isNaN(num)) {
                            if (drawingTool === 'brush') setBrushSize(num);
                            else if (drawingTool === 'highlighter') setHighlighterSize(num);
                            else if (drawingTool === 'eraser') setEraserSize(num);
                            setTempSizeValue(num.toString());
                          } else {
                            setTempSizeValue(getCurrentSize().toString());
                          }
                        }}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter') {
                            e.currentTarget.blur();
                          } else if (e.key === 'Escape') {
                            setTempSizeValue(getCurrentSize().toString());
                            e.currentTarget.blur();
                          }
                        }}
                      />
                    </div>
                  </div>
                  {/* Smoothing Slider (only for brush) */}
                  {drawingTool === 'brush' && (
                    <div className="flex flex-col gap-y-0.5">
                      <label className="text-brand-light text-[11px] font-medium">Smoothing</label>
                      <div className="flex items-center gap-x-2">
                        <Slider
                          className="w-full dark"
                          value={[smoothing * 100]}
                          max={100}
                          min={0}
                          step={1}
                          onValueChange={(value) => setSmoothing(value[0] / 100)}
                        />
                        <div className="w-[40px] h-6 px-1 text-brand-light border border-brand-light/10 text-center bg-brand text-[11px] font-normal flex items-center justify-center rounded">
                          {Math.round(smoothing * 100)}
                        </div>
                      </div>
                    </div>
                  )}
                  {/* Color Picker (hidden for eraser) */}
                  {drawingTool === 'brush' && (
                    <ColorInput
                      size="medium"
                      label="Color"
                      labelClass="text-brand-light text-[11px] font-medium"
                      value={drawingColor}
                      onChange={setDrawingColor}
                      percentValue={drawingOpacity}
                      setPercentValue={setDrawingOpacity}
                    />
                  )}
                </div>
              </PopoverContent>
            </Popover>
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
                 className='dark text-[11px] flex items-center gap-x-2 h-7 font-medium'
                 onClick={() => { setShape('rectangle'); setTool('shape'); setShapeOpen(false); }}
               >
                 <LuSquare className='w-4 h-4' />
                 <span>Rectangle</span>
                 {shape === 'rectangle' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
               </DropdownMenuItem>
               <DropdownMenuItem
                 key='shape-ellipse'
                 textValue='Ellipse'
                 className='dark text-[11px] flex items-center gap-x-2 h-7 font-medium'
                 onClick={() => { setShape('ellipse'); setTool('shape'); setShapeOpen(false); }}
               >
                 <LuCircle className='w-4 h-4' />
                 <span>Ellipse</span>
                 {shape === 'ellipse' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
               </DropdownMenuItem>
               <DropdownMenuItem
                 key='shape-polygon'
                 textValue='Polygon'
                 className='dark text-[11px] flex items-center gap-x-2 h-7 font-medium'
                 onClick={() => { setShape('polygon'); setTool('shape'); setShapeOpen(false); }}
               >
                 <LuTriangle className='w-4 h-4' />
                 <span>Polygon</span>
                 {shape === 'polygon' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
               </DropdownMenuItem>
               <DropdownMenuItem
                 key='shape-star'
                 textValue='Star'
                 className='dark text-[11px] flex items-center gap-x-2 h-6 font-medium'
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
                 className='dark text-[11px] flex items-center gap-x-2 h-7 font-medium'
                 onClick={() => { setTool('pointer'); setToolOpen(false); }}
               >
                 <LuMousePointer2 className='w-4 h-4' />
                 <span>Pointer</span>
                 {tool === 'pointer' && <LuCheck className='w-4 h-4 ml-auto text-brand-light' />}
               </DropdownMenuItem>
               
               <DropdownMenuItem
                 key='tool-hand'
                 textValue='Hand'
                 className='dark text-[11px] flex items-center gap-x-2 h-7 font-medium'
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