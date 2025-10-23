import { useManifest } from '@/lib/manifest/hooks';
import { useManifestStore } from '@/lib/manifest/store';
import React, { useState } from 'react'
import { LuChevronLeft, LuChevronDown, LuChevronRight, LuDownload, LuCheck } from 'react-icons/lu';
import type { ManifestComponent, ManifestComponentModelPathItem } from '@/lib/manifest/api';
import { ScrollArea } from '../ui/scroll-area';
import { cn } from '@/lib/utils';

interface ModelPageProps {
  manifestId: string;
}

const getComponentTypeLabel = (type: string): string => {
  const labels: Record<string, string> = {
    'transformer': 'Transformer',
    'text_encoder': 'Text Encoder',
    'vae': 'Variational Autoencoder',
    'scheduler': 'Scheduler',
    'helper': 'Helper'
  };
  return labels[type] || type.charAt(0).toUpperCase() + type.slice(1);
};

const formatComponentName = (name: string): string => {
  return name
    .replace(/\./g, ' ')
    .replace(/_/g, ' ')
    .replace(/-/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

const ComponentCard: React.FC<{ component: ManifestComponent }> = ({ component }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const modelPaths = Array.isArray(component.model_path) ? component.model_path : component.model_path ? [{ path: component.model_path }] : [];

  const handleDownload = (path: string) => {
    console.log('Download triggered for:', path);
  };

  // Check if component is downloaded (placeholder logic - replace with actual check)
  const isDownloaded = component.downloaded || false;

  const typeLabel = getComponentTypeLabel(component.type);
  const displayName = component.label || (component.name ? formatComponentName(component.name) : component.base ? formatComponentName(component.base) : typeLabel);

  // Check if there's any content to show when expanded
  const hasModelPaths = modelPaths.length > 0;
  const hasSchedulerOptions = component.type === 'scheduler' && component.scheduler_options && component.scheduler_options.length > 0;
  const hasExpandableContent = hasModelPaths || hasSchedulerOptions;

  return (
    <div className="bg-brand border border-brand-light/10 rounded-md text-start">
      {hasExpandableContent ? (
        <button 
          onClick={() => setIsExpanded(!isExpanded)}
          className="w-full flex items-center justify-between p-3 hover:bg-brand-background/30 transition-colors"
        >
          <div className="flex items-center gap-x-2 justify-between w-full mr-2">
            <div className="flex items-center gap-x-2">
              <div className={cn(
                "flex items-center justify-center w-5 h-5 rounded-full border",
                isDownloaded 
                  ? "bg-green-500/20 border-green-500/40" 
                  : "bg-brand-background border-brand-light/20"
              )}>
                {isDownloaded ? (
                  <LuCheck className="w-3 h-3 text-green-400" />
                ) : (
                  <LuDownload className="w-2.5 h-2.5 text-brand-light/50" />
                )}
              </div>
              <span className="text-brand-light text-[12px] font-medium">{displayName}</span>
            </div>
            <span className="text-brand-light/60 text-[10px] font-mono bg-brand-background px-2 py-0.5 rounded">{typeLabel}</span>
          </div>
          {isExpanded ? <LuChevronDown className="w-4 h-4 text-brand-light/60" /> : <LuChevronRight className="w-4 h-4 text-brand-light/60" />}
        </button>
      ) : (
        <div className="w-full flex items-center justify-between p-3">
          <div className="flex items-center gap-x-2 justify-between w-full">
            <div className="flex items-center gap-x-2">
              <div className={cn(
                "flex items-center justify-center w-5 h-5 rounded-full border",
                isDownloaded 
                  ? "bg-green-500/20 border-green-500/40" 
                  : "bg-brand-background border-brand-light/20"
              )}>
                {isDownloaded ? (
                  <LuCheck className="w-3 h-3 text-green-400" />
                ) : (
                  <LuDownload className="w-2.5 h-2.5 text-brand-light/50" />
                )}
              </div>
              <span className="text-brand-light text-[12px] font-medium">{displayName}</span>
            </div>
            <span className="text-brand-light/60 text-[10px] font-mono bg-brand-background px-2 py-0.5 rounded">{typeLabel}</span>
          </div>
        </div>
      )}
      
      {hasExpandableContent && isExpanded && (
        <div className="px-4 pb-4">

          {modelPaths.length > 0 && (
            <div className="space-y-2 mt-3">
              {modelPaths.map((item, idx) => {
                const pathItem = typeof item === 'string' ? { path: item } : item as ManifestComponentModelPathItem;
                return (
                  <div key={idx} className="bg-brand-background border border-brand-light/10 rounded-md  p-3">
                    <div className="flex items-start justify-between gap-x-2 ">
                      <div className="flex-1 min-w-0 flex-row items-center gap-x-2">
                        <div className="text-brand-light text-[10.5px] font-medium mb-1">Model Path</div>
                        <div className="text-brand-light text-[10px] font-mono break-all">{pathItem.path}</div>
                      </div>
                    </div>
                     {(pathItem.type || pathItem.precision) && (
                      <div className="flex flex-col items-start  mt-2 justify-start border-t border-brand-light/5  pt-2">
                      <h4 className="text-brand-light text-[10.5px] font-medium mb-1">
                        Model specifications
                      </h4>
                      {pathItem.type && (
                        <div className="text-[10px] flex flex-row items-center gap-x-1 ">
                          <span className="text-brand-light/60 font-medium">Model Type </span>
                          <span className="text-brand-light/80 font-mono">{pathItem.type === 'gguf' ? 'GGUF' : formatComponentName(pathItem.type)}</span>
                        </div>
                      )}
                
                    {pathItem.precision && (
                        <div className="text-[10px] flex flex-row items-center gap-x-1 ">
                          <span className="text-brand-light/60 font-medium">Precision </span>
                          <span className="text-brand-light/90 font-mono">{pathItem.precision.toUpperCase()}</span>
                        </div>
                      )}
                    </div>  
                    )}

                    {pathItem.resource_requirements && (
                      <div className="mt-2 pt-2 border-t border-brand-light/5">
                        <div className="text-brand-light text-[10.5px] font-medium mb-1">Resource Requirements</div>
                        <div className="flex flex-col ">
                          {pathItem.resource_requirements.min_vram_gb && (
                            <div className="text-[10px]">
                              <span className="text-brand-light/60 font-medium">Min VRAM </span>
                              <span className="text-brand-light/90">{pathItem.resource_requirements.min_vram_gb}GB</span>
                            </div>
                          )}
                          {pathItem.resource_requirements.recommended_vram_gb && (
                            <div className="text-[10px]">
                              <span className="text-brand-light/60 font-medium">Recommended VRAM </span>
                              <span className="text-brand-light/90">{pathItem.resource_requirements.recommended_vram_gb}GB</span>
                            </div>
                          )}
                          {pathItem.resource_requirements.compute_capability && (
                            <div className="text-[10px]">
                              <span className="text-brand-light/60">Compute Capability: </span>
                              <span className="text-brand-light/90">{pathItem.resource_requirements.compute_capability}</span>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    <button
                      onClick={() => handleDownload(pathItem.path)}
                      className="w-full mt-3 text-[10.5px] font-medium flex items-center justify-center gap-x-1.5 text-brand-light hover:text-brand-light/90 bg-brand hover:bg-brand/80 border border-brand-light/10 rounded-md px-3 py-2 transition-all"
                    >
                      <LuDownload className="w-3.5 h-3.5" />
                      <span>Download Model</span>
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          {component.type === 'scheduler' && component.scheduler_options && component.scheduler_options.length > 0 && (
            <div className="mt-3">
              <div className="text-brand-light/80 text-[11px] font-medium mb-2">Scheduler Options</div>
              <div className="space-y-2">
                {component.scheduler_options.map((option, idx) => (
                  <div key={idx} className="bg-brand-background border border-brand-light/5 rounded-[6px] px-3 py-1.5">
                    <div className="mb-1">
                      <span className="text-brand-light text-[11px] font-medium">{option.label || option.name}</span>
                    </div>
                    {option.description && (
                      <div className="text-[10px] text-brand-light/70 mt-1 mb-1.5">
                        {option.description}
                      </div>
                    )}
                    
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const ModelPage:React.FC<ModelPageProps> = ({ manifestId }) => {
  const { clearSelectedManifestId, getLoadedManifest } = useManifestStore();
  const simpleManifest = getLoadedManifest(manifestId);
  const { data: manifest } = useManifest(manifestId);

  const isVideoDemo = React.useMemo(() => {
    const value = (simpleManifest?.demo_path || '').toLowerCase();
    try {
      const url = new URL(value);
      const pathname = url.pathname;
      const ext = pathname.split('.').pop() || '';
      return ['mp4', 'webm', 'mov', 'm4v', 'ogg', 'm3u8'].includes(ext);
    } catch {
      return value.endsWith('.mp4') || value.endsWith('.webm') || value.endsWith('.mov') || value.endsWith('.m4v') || value.endsWith('.ogg') || value.endsWith('.m3u8');
    }
  }, [simpleManifest?.demo_path]);

  const components = manifest?.spec?.components || [];

  
  if (!manifest) return null;

  return ( 
    <div className="flex flex-col h-full w-full">
      <ScrollArea className="flex-1">
        <div className="p-7 pt-3 pb-28">
          <div className="flex items-center gap-x-3">
            <button onClick={clearSelectedManifestId} className="text-brand-light hover:text-brand-light/70 p-1 flex items-center justify-center bg-brand border border-brand-light/10 rounded transition-colors cursor-pointer">
              <LuChevronLeft className="w-3 h-3" />
            </button>
            <span className="text-brand-light/90 text-[11px] font-medium">Back</span>
          </div>
          
          <div className='mt-4 flex flex-row gap-x-4 w-full'>
            <div className="rounded-md overflow-hidden flex items-center w-44 aspect-square justify-start flex-shrink-0">
              {isVideoDemo ? (
                <video
                  src={manifest.metadata.demo_path}
                  className="h-full w-full object-cover rounded-md"
                  autoPlay
                  muted
                  loop
                  playsInline
                />
              ) : (
                <img src={manifest.metadata.demo_path} alt={manifest.metadata.name} className="h-full object-cover rounded-md" />
              )}
            </div>
            
            <div className='flex flex-col gap-y-1 w-full justify-start'>
              <h2 className="text-brand-light text-[18px] font-semibold text-start">{manifest.metadata.name}</h2>
              <p className="text-brand-light/90 text-[12px] text-start">{manifest.metadata.description}</p>
              
              <div className='flex flex-col mt-1 items-start gap-y-0.5'>
                <span className="text-brand-light text-[12px] font-medium">{manifest.metadata.license}</span>
                <span className="text-brand-light/80 text-[11px]">{manifest.metadata.author}</span>
              </div>
              
              <div className='flex flex-row items-center gap-x-1.5 mt-2'>
                {manifest.metadata?.tags?.map((tag) => (
                  <span key={tag} className="text-brand-light text-[11px] bg-brand-background border shadow border-brand-light/10 rounded px-2 py-0.5">{tag}</span>
                ))}
              </div>
            </div>
          </div>

            <div className="mt-6">
              <h3 className="text-brand-light text-[13.5px] font-semibold text-start">
                Model Architecture
              </h3>
              
              <div className="space-y-2 mt-3.5">
              {components.map((component, index) => (
                <ComponentCard key={index} component={component} />
              ))}
              {components.length === 0 && (
                <div className="text-brand-light/60 text-[12px] text-center py-8">No components available</div>
              )}
            </div>
          </div>
        </div>
      </ScrollArea>
    </div>
  )
}

export default ModelPage;