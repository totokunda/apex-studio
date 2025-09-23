import { MediaInfo, VideoClipProps } from '@/lib/types'
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {Image, Transformer, Group, Line} from 'react-konva'
import {fetchVideoSample, prefetchVideoSamples, setIdleDecodeTarget, removeIdleDecodeTarget} from '@/lib/media/video'
import {getMediaInfo} from '@/lib/media/utils'
import { useControlsStore } from '@/lib/control';
import Konva from 'konva';
import { useViewportStore } from '@/lib/viewport';
import { useClipStore } from '@/lib/clip';

const usePrefetch = (mediaInfo: MediaInfo | null, framesToPrefetch: number, currentFrame: number, path: string) => {
    const {codedWidth = 0, codedHeight = 0} = mediaInfo?.video || {};
    useEffect(() => {
      if (!mediaInfo) return;
      if (!Number.isFinite(currentFrame)) return;
      if (framesToPrefetch <= 0) return;
      (async () => {
        const secondDuration = mediaInfo.duration ?? 1;
        const packetRate = mediaInfo.stats.video?.averagePacketRate ?? 1;
        const totalFrames = secondDuration*packetRate;
        const center = Math.max(0, Math.min(totalFrames, Math.floor(currentFrame)));
        const half = Math.floor(framesToPrefetch / 2);
        const minFrame = Math.max(0, center - half);
        const maxFrame = Math.min(totalFrames, center + half);
        const count = maxFrame - minFrame + 1;
        if (count <= 0) return;
        const frameIndices = Array.from({length: count}, (_, i) => minFrame + i);
        try {
          await prefetchVideoSamples(path, frameIndices, codedWidth, codedHeight, {mediaInfo});
        } catch (e) {
          console.error(e);
        }
      })();
    }, [currentFrame, framesToPrefetch, mediaInfo, path, codedWidth, codedHeight]);
}

const VideoPreview: React.FC<VideoClipProps & {framesToPrefetch?: number, rectWidth: number, rectHeight: number}> = ({ src, clipId, startFrame = 0, framesToPrefetch = 96, rectWidth, rectHeight, framesToGiveStart}) => {
    const [mediaInfo, setMediaInfo] = useState<MediaInfo | null>(null);
    const focusFrame = useControlsStore((state) => state.focusFrame);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const imageRef = useRef<Konva.Image>(null);
    const transformerRef = useRef<Konva.Transformer>(null);
    const drawTokenRef = useRef(0);
    const suppressUntilRef = useRef<number>(0);
    const currentFrame = useMemo(() => focusFrame - startFrame + (framesToGiveStart || 0), [focusFrame, startFrame, framesToGiveStart]);
    const tool = useViewportStore((s) => s.tool);
    const scale = useViewportStore((s) => s.scale);
    const position = useViewportStore((s) => s.position);
    const setClipTransform = useClipStore((s) => s.setClipTransform);
    const clipTransform = useClipStore((s) => s.getClipTransform(clipId));
    const removeClipSelection = useControlsStore((s) => s.removeClipSelection);
    const addClipSelection = useControlsStore((s) => s.addClipSelection);
    const {selectedClipIds} = useControlsStore();
    const isSelected = useMemo(() => selectedClipIds.includes(clipId), [clipId, selectedClipIds]);

    const aspectRatio = useMemo(() => {
      const originalWidth = mediaInfo?.video?.codedWidth || 0;
      const originalHeight = mediaInfo?.video?.codedHeight || 0;
      if (!originalWidth || !originalHeight) return 16/9;
      const aspectRatio = originalWidth / originalHeight;
      return aspectRatio;
    }, [mediaInfo?.video?.codedWidth, mediaInfo?.video?.codedHeight]);



    const groupRef = useRef<Konva.Group>(null);
    const SNAP_THRESHOLD_PX = 4; // pixels at screen scale
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

    const updateGuidesAndMaybeSnap = useCallback((opts: { snap: boolean }) => {
        if (isRotating) return; // disable guides/snapping while rotating
        const node = imageRef.current;
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
        if (isRotating) return newBox; // do not snap bounds while rotating
        // Convert absolute newBox to local coordinates of the content group (rect space)
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

        // Snap left edge to 0, 25%, 75%
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
        // Snap right edge to rectWidth, 75%, 25%
        if (Math.abs(rectWidth - right) <= thresholdLocal) {
            local.width = rectWidth - local.x;
        } else if (Math.abs(v75 - right) <= thresholdLocal) {
            local.width = v75 - local.x;
        } else if (Math.abs(v25 - right) <= thresholdLocal) {
            local.width = v25 - local.x;
        }
        // Snap top edge to 0, 25%, 75%
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
        // Snap bottom edge to rectHeight, 75%, 25%
        if (Math.abs(rectHeight - bottom) <= thresholdLocal) {
            local.height = rectHeight - local.y;
        } else if (Math.abs(h75 - bottom) <= thresholdLocal) {
            local.height = h75 - local.y;
        } else if (Math.abs(h25 - bottom) <= thresholdLocal) {
            local.height = h25 - local.y;
        }

        // Convert back to absolute space
        let adjusted = {
            ...newBox,
            x: position.x + local.x * scale,
            y: position.y + local.y * scale,
            width: local.width * scale,
            height: local.height * scale,
        };


        // Prevent negative or zero sizes in absolute space just in case
        const MIN_SIZE_ABS = 1e-3;
        if (adjusted.width < MIN_SIZE_ABS) adjusted.width = MIN_SIZE_ABS;
        if (adjusted.height < MIN_SIZE_ABS) adjusted.height = MIN_SIZE_ABS;

        return adjusted;
    }, [rectWidth, rectHeight, scale, position.x, position.y, isRotating, aspectRatio]);

    // Create canvas once
    useEffect(() => {
        if (!canvasRef.current) {
            canvasRef.current = document.createElement('canvas');
        }
        return () => {
            canvasRef.current = null;
        };
    }, []);

    useEffect(() => {
        if (!isSelected) return;
        const tr = transformerRef.current;
        const img = imageRef.current;
        if (!tr || !img) return;
        const raf = requestAnimationFrame(() => {
            tr.nodes([img]);
            if (typeof (tr as any).forceUpdate === 'function') {
                (tr as any).forceUpdate();
            }
            tr.getLayer()?.batchDraw?.();
        });
        return () => cancelAnimationFrame(raf);
    }, [isSelected]);

    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                const info = await getMediaInfo(src);
                if (!cancelled) setMediaInfo(info);
            } catch (e) {
                console.error(e);
            }
        })();
        return () => { cancelled = true };
    }, [src]);

    // Compute aspect-fit display size and offsets within the preview rect
    const {displayWidth, displayHeight, offsetX, offsetY} = useMemo(() => {
        const originalWidth = mediaInfo?.video?.codedWidth || 0;
        const originalHeight = mediaInfo?.video?.codedHeight || 0;
        if (!originalWidth || !originalHeight || !rectWidth || !rectHeight) {
            return { displayWidth: 0, displayHeight: 0, offsetX: 0, offsetY: 0 };
        }
        const aspectRatio = originalWidth / originalHeight;
        let dw = rectWidth;
        let dh = rectHeight;
        if (rectWidth / rectHeight > aspectRatio) {
            dw = rectHeight * aspectRatio;
        } else {
            dh = rectWidth / aspectRatio;
        }
        const ox = (rectWidth - dw) / 2;
        const oy = (rectHeight - dh) / 2;
        return { displayWidth: dw, displayHeight: dh, offsetX: ox, offsetY: oy };
    }, [mediaInfo?.video?.codedWidth, mediaInfo?.video?.codedHeight, rectWidth, rectHeight]);

    // Initialize default transform if missing
    useEffect(() => {
        if (!clipTransform && displayWidth && displayHeight) {
            setClipTransform(clipId, { x: offsetX, y: offsetY, width: displayWidth, height: displayHeight, scaleX: 1, scaleY: 1, rotation: 0 });
        }
    }, [clipTransform, displayWidth, displayHeight, offsetX, offsetY, clipId, setClipTransform]);

    // Ensure canvas matches display size for crisp rendering
    useEffect(() => {
        if (!canvasRef.current) return;
        if (!displayWidth || !displayHeight) return;
        const canvas = canvasRef.current;
        const w = Math.floor(displayWidth);
        const h = Math.floor(displayHeight);
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
        }
    }, [displayWidth, displayHeight]);

    const draw = useCallback(async () => {
        if (!canvasRef.current) return;
        if (!mediaInfo) return;
        if (!displayWidth || !displayHeight) return;

        try {
            const secondDuration = mediaInfo.duration ?? 1;
            const packetRate = mediaInfo.stats.video?.averagePacketRate ?? 1;
            const totalFrames = secondDuration*packetRate;
            const targetFrame = Math.max(0, Math.min(totalFrames, Math.floor(currentFrame)));

            const myToken = ++drawTokenRef.current;
            const sample = await fetchVideoSample(src, targetFrame, mediaInfo?.video?.codedWidth, mediaInfo?.video?.codedHeight, {mediaInfo});
            if (!sample) return;
            if (myToken !== drawTokenRef.current) return;

            const canvas = canvasRef.current;
            if (!canvas) return;
            const ctx = canvas.getContext('2d');
            if (!ctx) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            sample.draw(ctx, 0, 0, canvas.width, canvas.height);
            imageRef.current?.getLayer()?.batchDraw?.();
        } catch (e) {
            console.error(e);
        }
    }, [mediaInfo, currentFrame, src, displayWidth, displayHeight]);

    usePrefetch(mediaInfo, framesToPrefetch, currentFrame, src);

  // Continuously decode frames when idle so the cache stays warm
  useEffect(() => {
      if (!mediaInfo || !mediaInfo.video) return;
      const { codedWidth = 0, codedHeight = 0 } = mediaInfo.video || {};
      if (!codedWidth || !codedHeight) return;
      if (!Number.isFinite(currentFrame)) return;
      setIdleDecodeTarget(src, Math.max(0, Math.floor(currentFrame)), codedWidth, codedHeight, { mediaInfo });
      return () => {
          removeIdleDecodeTarget(src);
      };
  }, [src, mediaInfo, currentFrame]);

    useEffect(() => {
        draw();
    }, [draw]);

    const handleDragMove = useCallback((e: Konva.KonvaEventObject<MouseEvent>) => {
        updateGuidesAndMaybeSnap({ snap: true });
        const node = imageRef.current;
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
        const persistTransform = () => {
            const node = imageRef.current;
            if (!node) return;
            setClipTransform(clipId, {
                x: node.x(),
                y: node.y(),
                width: node.width(),
                height: node.height(),
                scaleX: node.scaleX(),
                scaleY: node.scaleY(),
                rotation: node.rotation(),
            });
        };
        const onTransform = () => {
            bumpSuppress();
            if (!isRotating) {
                updateGuidesAndMaybeSnap({ snap: false });
            }
            persistTransform();
        };
        const onTransformEnd = () => {
            bumpSuppress();
            setIsTransforming(false);
            setIsInteracting(false);
            setIsRotating(false);
            setGuides({ vCenter: false, hCenter: false, v25: false, v75: false, h25: false, h75: false, left: false, right: false, top: false, bottom: false });
            persistTransform();
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
            const now = (typeof performance !== 'undefined' && performance.now) ? performance.now() : Date.now();
            if (now < suppressUntilRef.current) return;
            const stage = imageRef.current?.getStage();
            const container = stage?.container();
            // check that node is inside container
            const node = e.target;
            if (!container?.contains(node as Node)) return;
            if (!stage || !container || !imageRef.current) return;
            const containerRect = container.getBoundingClientRect();
            const pointerX = e.clientX - containerRect.left;
            const pointerY = e.clientY - containerRect.top;
            const imgRect = imageRef.current.getClientRect({ skipShadow: true, skipStroke: true });
            const insideImage = pointerX >= imgRect.x && pointerX <= imgRect.x + imgRect.width && pointerY >= imgRect.y && pointerY <= imgRect.y + imgRect.height;
            
            if (!insideImage) {
                removeClipSelection(clipId);
            }
        };
        window.addEventListener('click', handleWindowClick);
        return () => {
            window.removeEventListener('click', handleWindowClick);
        };
    }, [clipId, isSelected, removeClipSelection]);

  return (
    <React.Fragment>
    <Group ref={groupRef} clipX={0} clipY={0} clipWidth={rectWidth} clipHeight={rectHeight}>
      <Image 
      draggable={tool === 'pointer' && !isTransforming } 
      ref={imageRef}  
      image={canvasRef.current || undefined}
       x={clipTransform?.x ?? offsetX} 
       y={clipTransform?.y ?? offsetY} 
       width={clipTransform?.width ?? displayWidth} 
       height={clipTransform?.height ?? displayHeight} 
       scaleX={clipTransform?.scaleX ?? 1}
       scaleY={clipTransform?.scaleY ?? 1}
       rotation={clipTransform?.rotation ?? 0}
       onDragMove={handleDragMove} 
       onDragStart={handleDragStart} 
       onDragEnd={handleDragEnd} 
       onClick={handleClick} 
       />
      {tool === 'pointer' && isSelected && isInteracting && !isRotating && (
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
    {tool === 'pointer' && isSelected && <Transformer 
        borderStroke='#AE81CE'
        anchorCornerRadius={8} 
        anchorStroke='#E3E3E3' 
        anchorStrokeWidth={1}
        borderStrokeWidth={2}
        rotationSnaps={[0, 45, 90, 135, 180, 225, 270, 315]} 
        boundBoxFunc={transformerBoundBoxFunc as any}
        ref={(node) => {
            transformerRef.current = node;
            if (node && imageRef.current) {
                node.nodes([imageRef.current]);
                if (typeof (node as any).forceUpdate === 'function') {
                    (node as any).forceUpdate();
                }
                node.getLayer()?.batchDraw?.();
            }
        }} 
        enabledAnchors={['top-left', 'bottom-right', 'top-right', 'bottom-left']} />}
    </React.Fragment>
  )
}

export default VideoPreview