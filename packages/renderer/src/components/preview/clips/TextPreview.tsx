import { TextClipProps } from '@/lib/types'
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {Text, Transformer, Group, Line} from 'react-konva'
import { Html } from 'react-konva-utils'
import { useControlsStore } from '@/lib/control';
import Konva from 'konva';
import { useViewportStore } from '@/lib/viewport';
import { useClipStore } from '@/lib/clip';

//@ts-ignore
Konva._fixTextRendering = true;

interface TextEditorProps {
    onClose: () => void;
    onChange: (text: string) => void;
    textNode: Konva.Text | null;
    isEditing: boolean;
    width: number;
    height: number;
    textTransform: string;
    transformerRef: React.RefObject<Konva.Transformer | null>;
}

const applyTextTransform = (text: string, textTransform: string) => {
        if (textTransform === 'uppercase') {
            return text.toUpperCase();
        }
        if (textTransform === 'lowercase') {
            return text.toLowerCase();
        }
        if (textTransform === 'capitalize') {
            // Capitalize each word
            return text.split(' ').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
        } 
        return text;
    }

const TextEditor: React.FC<TextEditorProps> = ({ onClose, onChange, textNode, isEditing, width, height, textTransform, transformerRef }) => {
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const lastClickTimeRef = useRef<number>(0);

    useEffect(() => {
        if (!textareaRef.current || !textNode) return;
    
        const textarea = textareaRef.current;
        const textPosition = textNode.position();
    
        const areaPosition = {
          x: textPosition.x - 3,
          y: textPosition.y - 3,
        };

        // Calculate clientRect inside useEffect to get current dimensions
        let clientRect = null;
        if (transformerRef.current) {
            const rect = transformerRef.current.getClientRect();
            clientRect = {
                width: rect.width,
                height: rect.height,
                x: rect.x,
                y: rect.y,
            };
        }

        // Match styles with the text node
        textarea.value = applyTextTransform(textNode.text(), textTransform);
        textarea.style.position = 'absolute';
        textarea.style.top = `${areaPosition.y}px`;
        textarea.style.left = `${areaPosition.x}px`;
        textarea.style.fontSize = `${textNode.fontSize()}px`;
        textarea.style.padding = '0px';
        textarea.style.margin = '0px';
        textarea.style.overflow = 'hidden';
        textarea.style.background = 'none';
        textarea.style.outline = 'none';
        textarea.style.resize = 'none';
        textarea.style.lineHeight = String(textNode.lineHeight());
        textarea.style.fontFamily = textNode.fontFamily();
        textarea.style.fontStyle = String(textNode.fontStyle()).split(' ')[0]; // Extract 'normal' or 'italic'
        textarea.style.fontWeight = String(textNode.fontStyle()).includes('bold') ? 'bold' : 'normal';
        textarea.style.textDecoration = String(textNode.textDecoration());
        textarea.style.width = `${clientRect?.width ?? width}px`;
        textarea.style.height = `${clientRect?.height ?? height}px`;
        
        // Apply stroke if enabled
        if (textNode.strokeEnabled && textNode.strokeEnabled()) {
            const strokeColor = String(textNode.stroke());
            const strokeWidth = textNode.strokeWidth();
            textarea.style.webkitTextStroke = `${strokeWidth}px ${strokeColor}`;
        } else {
            textarea.style.webkitTextStroke = '';
        }
        
        // Apply shadow if enabled
        if (textNode.shadowEnabled && textNode.shadowEnabled()) {
            const shadowColor = String(textNode.shadowColor());
            const shadowBlur = textNode.shadowBlur();
            const shadowOffsetX = textNode.shadowOffsetX();
            const shadowOffsetY = textNode.shadowOffsetY();
            textarea.style.textShadow = `${shadowOffsetX}px ${shadowOffsetY}px ${shadowBlur}px ${shadowColor}`;
        } else {
            textarea.style.textShadow = '';
        }
        
        textarea.style.transformOrigin = 'left top';
        textarea.style.textAlign = textNode.align();
        textarea.style.verticalAlign = textNode.verticalAlign();
        textarea.style.color = String(textNode.fill());
    
        const rotation = textNode.rotation();
        let transform = '';
        if (rotation) {
          transform += `rotateZ(${rotation}deg)`;
        }
        textarea.style.transform = transform;
    
        textarea.focus();
    

      }, [textNode, onChange, onClose, textTransform, width, height]);

      useEffect(() => {
        const textarea = textareaRef.current;
        const DOUBLE_CLICK_THRESHOLD = 400;

        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
              onClose();
            }
          };
  
        const handleClickOutside = (e: MouseEvent) => {
              const target = e.target as Node;
              const currentTime = Date.now();
              const timeSinceLastClick = currentTime - lastClickTimeRef.current;
              lastClickTimeRef.current = currentTime;
              if (textarea === target || textarea?.contains(target) || !isEditing) {
                  return;
              }


              if (timeSinceLastClick < DOUBLE_CLICK_THRESHOLD) {
                  return;
              }

              onClose();
          }

        textarea?.addEventListener('keydown', handleKeyDown);
        document.addEventListener('click', handleClickOutside, true);
        
        return () => {
            document.removeEventListener('click', handleClickOutside, true);
            textarea?.removeEventListener('keydown', handleKeyDown);
        }
        
      }, [isEditing, onClose])

      return (
        <Html>
          <textarea
            onClick={e => e.stopPropagation()}
            onBlur={() => onChange(applyTextTransform(textareaRef.current?.value ?? '', textTransform))}
            onChange={(e) => onChange(applyTextTransform(e.target.value, textTransform))}
            ref={textareaRef}
            style={{
              border: '3px solid #AE81CE',
              position: 'absolute',
              display: isEditing ? 'block' : 'none',
            }}
          />
        </Html>
      );
};

const TextPreview: React.FC<TextClipProps & {rectWidth: number, rectHeight: number}> = ({ clipId, rectWidth, rectHeight}) => {
    const textRef = useRef<Konva.Text>(null);
    const transformerRef = useRef<Konva.Transformer>(null);
    const suppressUntilRef = useRef<number>(0);
    const tool = useViewportStore((s) => s.tool);
    const scale = useViewportStore((s) => s.scale);
    const position = useViewportStore((s) => s.position);
    const setClipTransform = useClipStore((s) => s.setClipTransform);
    const updateClip = useClipStore((s) => s.updateClip);
    const clipTransform = useClipStore((s) => s.getClipTransform(clipId));
    const clip = useClipStore((s) => s.getClipById(clipId)) as TextClipProps;
    const removeClipSelection = useControlsStore((s) => s.removeClipSelection);
    const addClipSelection = useControlsStore((s) => s.addClipSelection);
    const {selectedClipIds} = useControlsStore();
    const isSelected = useMemo(() => selectedClipIds.includes(clipId), [clipId, selectedClipIds]);

    const [isEditing, setIsEditing] = useState(false);

    const debounceTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const [temporaryText, setTemporaryText] = useState<string | null>(clip?.text ?? null);

    const groupRef = useRef<Konva.Group>(null);
    const SNAP_THRESHOLD_PX = 4;
    const [guides, setGuides] = useState({
        vCenter: false,
        hCenter: false,
        v25: false,
        v75: false,
        h25: false,
        h75: false,
        left: false,
        right: false,
        top: false,
        bottom: false,
    });

    const [isInteracting, setIsInteracting] = useState(false);
    const [isRotating, setIsRotating] = useState(false);
    const [isTransforming, setIsTransforming] = useState(false);

    

    const textTransform = clip?.textTransform ?? 'none';

    const text = clip?.text ?? 'Double-click to edit';
    const fontSize = clip?.fontSize ?? 48;
    const fontWeight = clip?.fontWeight ?? 400;
    const fontStyle = clip?.fontStyle ?? 'normal';
    const fontFamily = clip?.fontFamily ?? 'Arial';
    const color = clip?.color ?? '#000000';
    const textAlign = clip?.textAlign ?? 'left';
    const verticalAlign = clip?.verticalAlign ?? 'top';
    const textDecoration = clip?.textDecoration ?? 'none';
    const colorOpacity = clip?.colorOpacity ?? 100;
    
    // Stroke properties
    const strokeEnabled = clip?.strokeEnabled ?? false;
    const stroke = clip?.stroke ?? '#000000';
    const strokeWidth = clip?.strokeWidth ?? 2;
    
    // Shadow properties
    const shadowEnabled = clip?.shadowEnabled ?? false;
    const shadowColor = clip?.shadowColor ?? '#000000';
    const shadowOpacity = clip?.shadowOpacity ?? 75;
    const shadowBlur = clip?.shadowBlur ?? 4;
    const shadowOffsetX = clip?.shadowOffsetX ?? 2;
    const shadowOffsetY = clip?.shadowOffsetY ?? 2;
    

    // Don't render if clip is not found
    if (!clip) {
        return null;
    }

    const updateGuidesAndMaybeSnap = useCallback((opts: { snap: boolean }) => {
        if (isRotating) return;
        const node = textRef.current;
        const group = groupRef.current;
        if (!node || !group) return;
        const thresholdLocal = SNAP_THRESHOLD_PX / Math.max(0.0001, scale);
        const client = node.getClientRect({ skipShadow: true, skipStroke: true, relativeTo: group as any });
        const centerX = client.x + client.width / 2;
        const centerY = client.y + client.height / 2;
        const dxToVCenter = (rectWidth / 2) - centerX;
        const dyToHCenter = (rectHeight / 2) - centerY;
        const dxToV25 = (rectWidth * 0.25) - centerX;
        const dxToV75 = (rectWidth * 0.75) - centerX;
        const dyToH25 = (rectHeight * 0.25) - centerY;
        const dyToH75 = (rectHeight * 0.75) - centerY;
        const distVCenter = Math.abs(dxToVCenter);
        const distHCenter = Math.abs(dyToHCenter);
        const distV25 = Math.abs(dxToV25);
        const distV75 = Math.abs(dxToV75);
        const distH25 = Math.abs(dyToH25);
        const distH75 = Math.abs(dyToH75);
        const distLeft = Math.abs(client.x - 0);
        const distRight = Math.abs((client.x + client.width) - rectWidth);
        const distTop = Math.abs(client.y - 0);
        const distBottom = Math.abs((client.y + client.height) - rectHeight);

        const nextGuides = {
            vCenter: distVCenter <= thresholdLocal,
            hCenter: distHCenter <= thresholdLocal,
            v25: distV25 <= thresholdLocal,
            v75: distV75 <= thresholdLocal,
            h25: distH25 <= thresholdLocal,
            h75: distH75 <= thresholdLocal,
            left: distLeft <= thresholdLocal,
            right: distRight <= thresholdLocal,
            top: distTop <= thresholdLocal,
            bottom: distBottom <= thresholdLocal,
        };
        setGuides(nextGuides);

        if (opts.snap) {
            let deltaX = 0;
            let deltaY = 0;
            if (nextGuides.vCenter) {
                deltaX += dxToVCenter;
            } else if (nextGuides.v25) {
                deltaX += dxToV25;
            } else if (nextGuides.v75) {
                deltaX += dxToV75;
            } else if (nextGuides.left) {
                deltaX += -client.x;
            } else if (nextGuides.right) {
                deltaX += rectWidth - (client.x + client.width);
            }
            if (nextGuides.hCenter) {
                deltaY += dyToHCenter;
            } else if (nextGuides.h25) {
                deltaY += dyToH25;
            } else if (nextGuides.h75) {
                deltaY += dyToH75;
            } else if (nextGuides.top) {
                deltaY += -client.y;
            } else if (nextGuides.bottom) {
                deltaY += rectHeight - (client.y + client.height);
            }
            if (deltaX !== 0 || deltaY !== 0) {
                node.x(node.x() + deltaX);
                node.y(node.y() + deltaY);
                setClipTransform(clipId, { x: node.x(), y: node.y() });
            }
        }
    }, [rectWidth, rectHeight, scale, setClipTransform, clipId, isRotating]);

    const transformerBoundBoxFunc = useCallback((_oldBox: any, newBox: any) => {
        if (isRotating) return newBox;
        const invScale = 1 / Math.max(0.0001, scale);
        const local = {
            x: (newBox.x - position.x) * invScale,
            y: (newBox.y - position.y) * invScale,
            width: newBox.width * invScale,
            height: newBox.height * invScale,
        };
        const thresholdLocal = SNAP_THRESHOLD_PX * invScale;

        const left = local.x;
        const right = local.x + local.width;
        const top = local.y;
        const bottom = local.y + local.height;
        const v25 = rectWidth * 0.25;
        const v75 = rectWidth * 0.75;
        const h25 = rectHeight * 0.25;
        const h75 = rectHeight * 0.75;

        if (Math.abs(left - 0) <= thresholdLocal) {
            local.x = 0;
            local.width = right - local.x;
        } else if (Math.abs(left - v25) <= thresholdLocal) {
            local.x = v25;
            local.width = right - local.x;
        } else if (Math.abs(left - v75) <= thresholdLocal) {
            local.x = v75;
            local.width = right - local.x;
        }
        if (Math.abs(rectWidth - right) <= thresholdLocal) {
            local.width = rectWidth - local.x;
        } else if (Math.abs(v75 - right) <= thresholdLocal) {
            local.width = v75 - local.x;
        } else if (Math.abs(v25 - right) <= thresholdLocal) {
            local.width = v25 - local.x;
        }
        if (Math.abs(top - 0) <= thresholdLocal) {
            local.y = 0;
            local.height = bottom - local.y;
        } else if (Math.abs(top - h25) <= thresholdLocal) {
            local.y = h25;
            local.height = bottom - local.y;
        } else if (Math.abs(top - h75) <= thresholdLocal) {
            local.y = h75;
            local.height = bottom - local.y;
        }
        if (Math.abs(rectHeight - bottom) <= thresholdLocal) {
            local.height = rectHeight - local.y;
        } else if (Math.abs(h75 - bottom) <= thresholdLocal) {
            local.height = h75 - local.y;
        } else if (Math.abs(h25 - bottom) <= thresholdLocal) {
            local.height = h25 - local.y;
        }

        let adjusted = {
            ...newBox,
            x: position.x + local.x * scale,
            y: position.y + local.y * scale,
            width: local.width * scale,
            height: local.height * scale,
        };

        const MIN_SIZE_ABS = 1e-3;
        if (adjusted.width < MIN_SIZE_ABS) adjusted.width = MIN_SIZE_ABS;
        if (adjusted.height < MIN_SIZE_ABS) adjusted.height = MIN_SIZE_ABS;

        return adjusted;
    }, [rectWidth, rectHeight, scale, position.x, position.y, isRotating]);

    useEffect(() => {
        if (!isSelected) return;
        const tr = transformerRef.current;
        const txt = textRef.current;
        if (!tr || !txt) return;
        const raf = requestAnimationFrame(() => {
            tr.nodes([txt]);
            if (typeof (tr as any).forceUpdate === 'function') {
                (tr as any).forceUpdate();
            }
            tr.getLayer()?.batchDraw?.();
        });
        return () => cancelAnimationFrame(raf);
    }, [isSelected]);

    // Initialize default transform if missing
    const defaultWidth = 400;
    const defaultHeight = 100;
    const offsetX = (rectWidth - defaultWidth) / 2;
    const offsetY = (rectHeight - defaultHeight) / 2;

    useEffect(() => {
        if (!clipTransform) {
            setClipTransform(clipId, { x: offsetX, y: offsetY, width: defaultWidth, height: defaultHeight, scaleX: 1, scaleY: 1, rotation: 0 });
        }
    }, [clipTransform, offsetX, offsetY, clipId, setClipTransform]);

    

    const handleDragMove = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        updateGuidesAndMaybeSnap({ snap: true });
        const node = textRef.current;
        if (node) {
            setClipTransform(clipId, { x: node.x(), y: node.y() });
        } else {
            setClipTransform(clipId, { x: e.target.x(), y: e.target.y() });
        }
    }, [setClipTransform, clipId, updateGuidesAndMaybeSnap]);

    const handleDragStart = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        e.target.getStage()!.container().style.cursor = 'grab';
        addClipSelection(clipId);
        const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 250);
        setIsInteracting(true);
        updateGuidesAndMaybeSnap({ snap: true });
    }, [clipId, addClipSelection, updateGuidesAndMaybeSnap]);

    const handleDragEnd = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        e.target.getStage()!.container().style.cursor = 'default';
        const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
        suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 250);
        setClipTransform(clipId, { x: e.target.x(), y: e.target.y() });
        setIsInteracting(false);
        setGuides({ vCenter: false, hCenter: false, v25: false, v75: false, h25: false, h75: false, left: false, right: false, top: false, bottom: false });
    }, [setClipTransform, clipId]);

    const handleClick = useCallback(() => {
        addClipSelection(clipId);
    }, [addClipSelection, clipId]);

    const handleDblClick = useCallback(() => {
        if (tool !== 'pointer') return;
        setIsEditing(true);
    }, [tool]);


    const handleTextEditChange = useCallback((text: string) => {
        setTemporaryText(text);
        updateClip(clipId, { text: text });
    }, [clipId, updateClip]);

    useEffect(() => {
        if (transformerRef.current && textRef.current) {
          transformerRef.current.nodes([textRef.current]);
        }
      }, [isEditing]);

    useEffect(() => {
        return () => {
            if (debounceTimeoutRef.current) {
                clearTimeout(debounceTimeoutRef.current);
            }
        };
    }, []);
    

   
    useEffect(() => {
        const transformer = transformerRef.current;
        if (!transformer) return;
        const bumpSuppress = () => {
            const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
            suppressUntilRef.current = Math.max(suppressUntilRef.current, now + 300);
        };
        const onTransformStart = () => {
            bumpSuppress();
            setIsTransforming(true);
            const active = (transformer as any)?.getActiveAnchor?.();
            const rotating = typeof active === 'string' && active.includes('rotater');
            setIsRotating(!!rotating);
            setIsInteracting(true);
            if (!rotating) {
                updateGuidesAndMaybeSnap({ snap: false });
            } else {
                setGuides({ vCenter: false, hCenter: false, v25: false, v75: false, h25: false, h75: false, left: false, right: false, top: false, bottom: false });
            }
        };

        const onTransform = () => {
            bumpSuppress();
            const node = textRef.current;
            if (!node) return;
            
            // Calculate new dimensions from scale
            const newWidth = node.width() * node.scaleX();
            const newHeight = node.height() * node.scaleY();
            
            // Immediately reset scale and apply as width/height instead
            // This prevents visual scaling of the text content
            node.width(newWidth);
            node.height(newHeight);
            node.scaleX(1);
            node.scaleY(1);
            
            setClipTransform(clipId, {
                x: node.x(),
                y: node.y(),
                width: newWidth,
                height: newHeight,
                scaleX: 1,
                scaleY: 1,
                rotation: node.rotation(),
            });
            
            if (!isRotating) {
                updateGuidesAndMaybeSnap({ snap: false });
            }
        };
        const onTransformEnd = () => {
            bumpSuppress();
            const node = textRef.current;
            if (node) {
                const newWidth = node.width() * node.scaleX();
                const newHeight = node.height() * node.scaleY();
                node.width(newWidth);
                node.height(newHeight);
                node.scaleX(1);
                node.scaleY(1);
                setClipTransform(clipId, {
                    x: node.x(),
                    y: node.y(),
                    width: newWidth,
                    height: newHeight,
                    scaleX: 1,
                    scaleY: 1,
                    rotation: node.rotation(),
                });
            }
            setIsTransforming(false);
            setIsInteracting(false);
            setIsRotating(false);
            setGuides({ vCenter: false, hCenter: false, v25: false, v75: false, h25: false, h75: false, left: false, right: false, top: false, bottom: false });
        };
        transformer.on('transformstart', onTransformStart);
        transformer.on('transform', onTransform);
        transformer.on('transformend', onTransformEnd);
        return () => {
            transformer.off('transformstart', onTransformStart);
            transformer.off('transform', onTransform);
            transformer.off('transformend', onTransformEnd);
        };
    }, [transformerRef.current, updateGuidesAndMaybeSnap, setClipTransform, clipId, isRotating]);

    useEffect(() => {
        const handleWindowClick = (e: MouseEvent) => {
            if (!isSelected) return;
            if (isEditing) return; // Don't deselect while editing
            const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
            if (now < suppressUntilRef.current) return;
            const stage = textRef.current?.getStage();
            const container = stage?.container();
            const node = e.target;
            if (!container?.contains(node as Node)) return;
            if (!stage || !container || !textRef.current) return;
            const containerRect = container.getBoundingClientRect();
            const pointerX = e.clientX - containerRect.left;
            const pointerY = e.clientY - containerRect.top;
            const textRect = textRef.current.getClientRect({ skipShadow: true, skipStroke: true });
            const insideText = pointerX >= textRect.x && pointerX <= textRect.x + textRect.width && pointerY >= textRect.y && pointerY <= textRect.y + textRect.height;
            if (!insideText) {
                removeClipSelection(clipId);
            }
        };
        window.addEventListener('click', handleWindowClick);
        return () => {
            window.removeEventListener('click', handleWindowClick);
        };
    }, [clipId, isSelected, removeClipSelection, isEditing]);

    useEffect(() => {
        if (!isEditing) {
            setTemporaryText(clip?.text ?? '');
        }
    }, [isEditing, clip?.text]);

  return (
    <React.Fragment>
    <Group ref={groupRef} clipX={0} clipY={0} clipWidth={rectWidth} clipHeight={rectHeight}>
      <Text 
      draggable={tool === 'pointer' && !isTransforming && !isEditing} 
      ref={textRef}
      text={isEditing ? applyTextTransform(temporaryText ?? text, textTransform) : applyTextTransform(text, textTransform)}
      fontSize={fontSize}
      fontFamily={fontFamily}
      fontStyle={`${fontStyle} ${fontWeight >= 700 ? 'bold' : 'normal'}`}
      textDecoration={textDecoration}
      fill={color}
      fillOpacity={(colorOpacity ?? 100) / 100}
      stroke={strokeEnabled ? stroke : undefined}
      strokeWidth={strokeEnabled ? strokeWidth : undefined}
      strokeEnabled={strokeEnabled}
      shadowColor={shadowEnabled ? shadowColor : undefined}
      shadowBlur={shadowEnabled ? shadowBlur : undefined}
      shadowOpacity={shadowEnabled ? (shadowOpacity ?? 100) / 100 : undefined}
      shadowOffsetX={shadowEnabled ? shadowOffsetX : undefined}
      shadowOffsetY={shadowEnabled ? shadowOffsetY : undefined}
      shadowEnabled={shadowEnabled}
      align={textAlign}
      verticalAlign={verticalAlign}
      visible={!isEditing}
      opacity={(clipTransform?.opacity ?? 100) / 100}
       x={clipTransform?.x ?? offsetX} 
       y={clipTransform?.y ?? offsetY} 
       width={clipTransform?.width ?? defaultWidth} 
       height={clipTransform?.height ?? defaultHeight} 
       scaleX={1}
       scaleY={1}
       rotation={clipTransform?.rotation ?? 0}
       onDragMove={handleDragMove} 
       onDragStart={handleDragStart} 
       onDragEnd={handleDragEnd} 
       onClick={handleClick} 
       onDblClick={handleDblClick}
       onTransform={(e) => {
         const node = e.target as Konva.Text;
         const newWidth = node.width() * node.scaleX();
         const newHeight = node.height() * node.scaleY();
         node.setAttrs({
           width: newWidth,
           height: newHeight,
           scaleX: 1,
           scaleY: 1,
         });
       }}
       />
       <TextEditor 
            onClose={() => setIsEditing(false)} 
            onChange={handleTextEditChange} 
            textNode={textRef.current}
            textTransform={textTransform}
            isEditing={isEditing}
            width={clipTransform?.width ?? defaultWidth}
            height={clipTransform?.height ?? defaultHeight}
            transformerRef={transformerRef}
       />
      {tool === 'pointer' && isSelected && isInteracting && !isRotating && !isEditing && (
        <React.Fragment>
          {guides.vCenter && <Line listening={false} points={[rectWidth/2, 0, rectWidth/2, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.v25 && <Line listening={false} points={[rectWidth*0.25, 0, rectWidth*0.25, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.v75 && <Line listening={false} points={[rectWidth*0.75, 0, rectWidth*0.75, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.hCenter && <Line listening={false} points={[0, rectHeight/2, rectWidth, rectHeight/2]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.h25 && <Line listening={false} points={[0, rectHeight*0.25, rectWidth, rectHeight*0.25]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.h75 && <Line listening={false} points={[0, rectHeight*0.75, rectWidth, rectHeight*0.75]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.left && <Line listening={false} points={[0, 0, 0, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.right && <Line listening={false} points={[rectWidth, 0, rectWidth, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.top && <Line listening={false} points={[0, 0, rectWidth, 0]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
          {guides.bottom && <Line listening={false} points={[0, rectHeight, rectWidth, rectHeight]} stroke={'#AE81CE'} strokeWidth={1} dash={[6, 4]} />}
        </React.Fragment>
      )}
    </Group>
    {tool === 'pointer' && isSelected && !isEditing && <Transformer 
        borderStroke='#AE81CE'
        anchorCornerRadius={8} 
        anchorStroke='#E3E3E3' 
        anchorStrokeWidth={1}
        borderStrokeWidth={2}
        rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]} 
        boundBoxFunc={transformerBoundBoxFunc as any}
        
        keepRatio={false}
        ref={(node) => {
            transformerRef.current = node;
            if (node && textRef.current) {
                node.nodes([textRef.current]);
                if (typeof (node as any).forceUpdate === 'function') {
                    (node as any).forceUpdate();
                }
                node.getLayer()?.batchDraw?.();
            }
        }} 
        enabledAnchors={['top-left', 'bottom-right', 'top-right', 'bottom-left',  'middle-left', 'middle-right', 'top-center', 'bottom-center']} />}
    </React.Fragment>
  )
}

export default TextPreview;

