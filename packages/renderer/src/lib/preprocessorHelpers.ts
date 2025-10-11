import { PreprocessorClipProps } from './types'

/**
 * Converts an x position on the timeline to a frame number
 */
export function calculateFrameFromX(
    xPosition: number,
    timelinePadding: number,
    timelineWidth: number,
    timelineDuration: [number, number]
): number {
    const timelineX = xPosition - timelinePadding;
    const [startFrame, endFrame] = timelineDuration;
    const framePosition = (timelineX / timelineWidth) * (endFrame - startFrame) + startFrame;
    return Math.round(framePosition);
}

/**
 * Gets all preprocessors for a clip excluding a specific one
 */
export function getOtherPreprocessors(
    allPreprocessors: PreprocessorClipProps[],
    excludeId: string
): PreprocessorClipProps[] {
    return allPreprocessors
        .filter(p => p.id !== excludeId)
        .sort((a, b) => (a.startFrame ?? 0) - (b.startFrame ?? 0));
}

/**
 * Detects collisions between a target range and other preprocessors
 */
export function detectCollisions(
    targetStart: number,
    targetEnd: number,
    otherPreprocessors: PreprocessorClipProps[],
    clipDuration: number
): PreprocessorClipProps[] {
    return otherPreprocessors.filter(p => {
        const pStart = p.startFrame ?? 0;
        const pEnd = p.endFrame ?? clipDuration;
        return !(targetEnd <= pStart || targetStart >= pEnd);
    });
}

/**
 * Finds a gap after a block of preprocessors where a preprocessor can fit
 */
export function findGapAfterBlock(
    collidingPreprocessors: PreprocessorClipProps[],
    direction: 'left' | 'right',
    preprocessorDuration: number,
    clipDuration: number
): number | null {
    if (collidingPreprocessors.length === 0) return null;

    const sorted = [...collidingPreprocessors].sort((a, b) => (a.startFrame ?? 0) - (b.startFrame ?? 0));
    
    let blockStart = sorted[0].startFrame ?? 0;
    let blockEnd = sorted[0].endFrame ?? clipDuration;
    
    for (let i = 1; i < sorted.length; i++) {
        const current = sorted[i];
        const currentStart = current.startFrame ?? 0;
        const currentEnd = current.endFrame ?? clipDuration;
        
        if (currentStart <= blockEnd) {
            blockEnd = Math.max(blockEnd, currentEnd);
        }
    }
    
    if (direction === 'right') {
        const gapStart = blockEnd;
        const gapEnd = clipDuration;
        const availableSpace = gapEnd - gapStart;
        
        if (availableSpace >= preprocessorDuration) {
            return gapStart;
        }
    } else {
        const gapStart = 0;
        const gapEnd = blockStart;
        const availableSpace = gapEnd - gapStart;
        
        if (availableSpace >= preprocessorDuration) {
            return gapEnd - preprocessorDuration;
        }
    }
    
    return null;
}

