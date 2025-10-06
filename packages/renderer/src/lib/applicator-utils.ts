import { useClipStore } from './clip';
import { useControlsStore } from './control';
import { AnyClipProps, ClipType, FilterClipProps } from './types';
import { BaseClipApplicator } from '@/components/preview/clips/apply/base';
import { FilterPreview } from '@/components/preview/clips/apply/filter';
import { WebGLHaldClut } from '@/components/preview/webgl-filters/hald-clut';

/**
 * Supported applicator clip types that can be applied to other clips
 */
export const APPLICATOR_CLIP_TYPES: ClipType[] = ['filter', 'mask', 'processor'];

/**
 * Gets all applicable effect clips that should be applied to a given clip at the current frame.
 * Returns clips from timelines above the target clip's timeline, in the correct order (bottom to top).
 * Only includes clips of types that can be applied as effects (filters, masks, processors, etc.)
 * 
 * @param clipId - The ID of the target clip to get applicators for
 * @param applicatorTypes - Optional array of clip types to filter by (defaults to all applicator types)
 * @returns Array of AnyClipProps sorted by timeline order (bottom to top for proper stacking)
 */
export function getApplicableClips(
    clipId: string, 
    applicatorTypes: ClipType[] = APPLICATOR_CLIP_TYPES
): AnyClipProps[] {
    const clipStore = useClipStore.getState();
    const controlStore = useControlsStore.getState();
    
    const clip = clipStore.getClipById(clipId);
    if (!clip) return [];
    
    const focusFrame = controlStore.focusFrame;
    const timelines = clipStore.timelines;
    const allClips = clipStore.clips;
    
    // Find the timeline index of the current clip
    const clipTimelineIndex = timelines.findIndex(t => t.timelineId === clip.timelineId);
    if (clipTimelineIndex === -1) return [];
    
    // Get all applicable clips from timelines with LOWER indices (rendered above)
    const applicableClips: AnyClipProps[] = [];
    
    for (let i = 0; i < clipTimelineIndex; i++) {
        const timeline = timelines[i];
        const timelineClips = allClips.filter(c => c.timelineId === timeline.timelineId);
        
        // Check if timeline is hidden
        if (timeline.hidden) continue;
        
        for (const timelineClip of timelineClips) {
            // Only include clips of applicable types
            if (!applicatorTypes.includes(timelineClip.type)) continue;
            
            // Check if the clip overlaps with the current frame
            const startFrame = timelineClip.startFrame ?? 0;
            const endFrame = timelineClip.endFrame ?? 0;
            const isInRange = focusFrame >= startFrame && focusFrame <= endFrame;
            
            if (isInRange) {
                applicableClips.push(timelineClip);
            }
        }
    }
    
    // Sort by timeline index (lower index = rendered on top, but we want to apply bottom-up)
    // So we reverse to apply effects from bottom to top
    applicableClips.sort((a, b) => {
        const indexA = timelines.findIndex(t => t.timelineId === a.timelineId);
        const indexB = timelines.findIndex(t => t.timelineId === b.timelineId);
        return indexB - indexA; // Higher index (lower layer) applies first
    });
    
    return applicableClips;
}

/**
 * Configuration object for applicator factory
 */
export interface ApplicatorFactoryConfig {
    haldClutInstance?: WebGLHaldClut | null;
    // Add more shared resources here as needed (e.g., maskProcessor, etc.)
}

/**
 * Factory function to create an applicator instance from a clip.
 * Extend this function to support new applicator types.
 * 
 * @param clip - The clip to create an applicator from
 * @param config - Shared resources and configuration for applicators
 * @returns BaseClipApplicator instance or null if clip type is not supported
 */
export function createApplicatorFromClip(
    clip: AnyClipProps,
    config: ApplicatorFactoryConfig
): BaseClipApplicator | null {
    switch (clip.type) {
        case 'filter': {
            // Don't create filter applicator if haldClutInstance is not available
            if (!config.haldClutInstance) {
                console.warn('[ApplicatorFactory] Cannot create FilterPreview: haldClutInstance is null');
                return null;
            }
            
            const filterApplicator = new FilterPreview(clip as FilterClipProps);
            filterApplicator.setHaldClutInstance(config.haldClutInstance);
            return filterApplicator;
        }
        
        // Add more applicator types here as they are implemented
        // case 'mask': {
        //     const maskApplicator = new MaskPreview(clip as MaskClipProps);
        //     if (config.maskProcessor) {
        //         maskApplicator.setMaskProcessor(config.maskProcessor);
        //     }
        //     return maskApplicator;
        // }
        
        // case 'processor': {
        //     return new ProcessorPreview(clip as ProcessorClipProps);
        // }
        
        default:
            console.warn(`[ApplicatorFactory] Unsupported applicator type: ${clip.type}`);
            return null;
    }
}

/**
 * Convenience function to get all applicators for a clip.
 * Combines getApplicableClips and createApplicatorFromClip.
 * 
 * @param clipId - The ID of the target clip
 * @param config - Shared resources and configuration for applicators
 * @param applicatorTypes - Optional array of clip types to filter by
 * @returns Array of BaseClipApplicator instances ready to be applied
 */
export function getApplicatorsForClip(
    clipId: string,
    config: ApplicatorFactoryConfig,
    applicatorTypes?: ClipType[]
): BaseClipApplicator[] {
    const applicableClips = getApplicableClips(clipId, applicatorTypes);
    
    const applicators = applicableClips
        .map(clip => createApplicatorFromClip(clip, config))
        .filter((applicator): applicator is BaseClipApplicator => applicator !== null);
    
    return applicators;
}

