import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { UIPanel, UIInput } from '@/lib/manifest/api'
import { ChevronDown } from 'lucide-react'
import { cn } from '@/lib/utils';
import TextInput from './inputs/TextInput';
import { useClipStore } from '@/lib/clip';
import type { IconType } from 'react-icons';
import SelectInput from './inputs/SelectInput';
import NumberInput from './inputs/NumberInput';
import NumberInputSlider from './inputs/NumberInputSlider';
import RandomInput from './inputs/RandomInput';
import BooleanInput from './inputs/BooleanInput';
import ImageInput from './inputs/ImageInput';

export const ModelInputsPanel: React.FC<{ panel: UIPanel, inputs: UIInput[], clipId: string, panelSize:number }> = ({ panel, inputs, clipId, panelSize }) => {

    const updateModelInput = useClipStore((s) => s.updateModelInput);
    const [iconEl, setIconEl] = useState<React.ReactNode>(null);

    const getInputById = useCallback((id: string) => {
        return inputs.find((input) => input.id === id) as UIInput;
    }, [inputs]);

    const collapsible = useMemo(() => {
        return panel.collapsible === true;
    }, [panel]);

    const [collapsed, setCollapsed] = useState(false);

    useEffect(() => {
      let cancelled = false;
      async function loadIcon() {
        if (!panel.icon) {
          setIconEl(null);
          return;
        }
        const [libraryRaw, iconName] = panel.icon.split('/');
        const library = libraryRaw?.toLowerCase();
        const loaders: Record<string, () => Promise<Record<string, IconType>>> = {
          lu: () => import('react-icons/lu') as unknown as Promise<Record<string, IconType>>,
          tfi: () => import('react-icons/tfi') as unknown as Promise<Record<string, IconType>>,
          md: () => import('react-icons/md') as unknown as Promise<Record<string, IconType>>,
          tb: () => import('react-icons/tb') as unknown as Promise<Record<string, IconType>>,
          fa: () => import('react-icons/fa') as unknown as Promise<Record<string, IconType>>,
          fa6: () => import('react-icons/fa6') as unknown as Promise<Record<string, IconType>>,
          ri: () => import('react-icons/ri') as unknown as Promise<Record<string, IconType>>,
          go: () => import('react-icons/go') as unknown as Promise<Record<string, IconType>>,
          sl: () => import('react-icons/sl') as unknown as Promise<Record<string, IconType>>,
          fi: () => import('react-icons/fi') as unknown as Promise<Record<string, IconType>>,
          cg: () => import('react-icons/cg') as unknown as Promise<Record<string, IconType>>,
          rx: () => import('react-icons/rx') as unknown as Promise<Record<string, IconType>>,
          io5: () => import('react-icons/io5') as unknown as Promise<Record<string, IconType>>,
          bs: () => import('react-icons/bs') as unknown as Promise<Record<string, IconType>>,
          hi: () => import('react-icons/hi') as unknown as Promise<Record<string, IconType>>,
          vsc: () => import('react-icons/vsc') as unknown as Promise<Record<string, IconType>>,
          bi: () => import('react-icons/bi') as unknown as Promise<Record<string, IconType>>,
          pi: () => import('react-icons/pi') as unknown as Promise<Record<string, IconType>>,
          io: () => import('react-icons/io') as unknown as Promise<Record<string, IconType>>,
        };
        const loader = loaders[library || ''];
        if (!loader || !iconName) {
          setIconEl(null);
          return;
        }
        try {
          const mod = await loader();
          const Exported = (mod as unknown as Record<string, any>)[iconName];
          if (!cancelled) {
            if (Exported) {
              let ComponentToRender: any | null = null;
              if (React.isValidElement(Exported)) {
                ComponentToRender = (Exported as React.ReactElement).type as any;
              } else if (typeof Exported === 'function' || (typeof Exported === 'object' && Exported !== null)) {
                ComponentToRender = Exported as any;
              }
              if (ComponentToRender) {
                setIconEl(React.createElement(ComponentToRender, { className: "h-3.5 w-3.5 text-brand-light/80" }));
              } else {
                setIconEl(null);
              }
            } else {
              setIconEl(null);
            }
          }
        } catch {
          if (!cancelled) setIconEl(null);
        }
      }
      loadIcon();
      return () => {
        cancelled = true;
      }
    }, [panel.icon]);

  return (
    <div className=" bg-brand shadow   rounded-[6px] border border-brand-light/5">
      <div onClick={() => setCollapsed((v) => !v)} className={cn("flex items-center justify-between bg-brand rounded-[6px]  py-2.5 px-3", {
        'rounded-b': collapsed,
        'rounded-b-none ': !collapsed || !collapsible,
        'cursor-pointer': collapsible,
      })}>
        <h3 className="text-brand-light text-[11.5px] text-start font-medium flex items-center gap-x-2">
          {iconEl}
          {panel.label}
        </h3>
        {collapsible && (
          <button
            type="button"
            aria-label={collapsed ? 'Expand panel' : 'Collapse panel'}
            
            className="p-1 text-brand-light/60 hover:text-brand-light transition-colors"
          >
            <ChevronDown className={`h-3.5 w-3.5 transition-transform ${collapsed ? '-rotate-90' : 'rotate-0'}`} />
          </button>
        )}
      </div>
      {(!collapsible || !collapsed) && (
      <div
        className='px-3 py-3 pb-5 pt-2'
        style={{
          display: 'flex',
          flexDirection: panel.layout.flow as 'row' | 'column',
          gap: '10px',
        }}>
        {panel.layout.rows.map((row) => {
          return (
            <div key={row.join('-')} style={{
              display: 'flex',
              flexDirection: 'row',
              gap: '10px',
            }}>
              {row.map((inputId) => {
                const input = getInputById(inputId);
                switch (input?.type) {
                  case 'text':
                    return <TextInput key={inputId} label={input?.label} description={input?.description} value={input?.value || ''} defaultValue={input?.default} onChange={(value) => updateModelInput(clipId, inputId, { value })} placeholder={input?.placeholder} />
                  case 'select':
                    return <SelectInput key={inputId} label={input?.label} defaultOption={input?.default} description={input?.description} value={input?.value || ''} onChange={(value) => updateModelInput(clipId, inputId, { value })} options={input?.options || []} />
                  case 'boolean': {
                      const boolVal = String(input?.value ?? input?.default ?? 'false').toLowerCase() === 'true';
                      return (
                        <BooleanInput
                          key={inputId}
                          label={input?.label}
                          description={input?.description}
                          value={boolVal}
                          onChange={(v) => updateModelInput(clipId, inputId, { value: v ? 'true' : 'false' })}
                        />
                      );
                    }
                  case 'number+slider': {
                      const numVal = Number(input?.value ?? input?.default ?? 0);
                      const toFixed = input.step ? input.step.toString().split('.')[1]?.length ?? 0 : 0;

                      return (
                        <NumberInputSlider
                          key={inputId}
                          label={input?.label || ''}
                          description={input?.description}
                          value={Number.isFinite(numVal) ? numVal : 0}
                          min={input?.min}
                          max={input?.max}
                          step={input?.step}
                          toFixed={toFixed}
                          onChange={(v) => updateModelInput(clipId, inputId, { value: v.toString() })}
                        />
                      )
                    }
                  
                  case 'number': {
                      const strVal = String(input?.value ?? input?.default ?? '');
                      return (
                        <NumberInput
                          startLogo={input?.label ? input?.label.charAt(0).toUpperCase() : ''}
                          key={inputId}
                          
                          label={input?.label}
                          description={input?.description}
                          value={strVal}
                          min={input?.min}      
                          max={input?.max}
                          step={input?.step}
                          onChange={(v) => updateModelInput(clipId, inputId, { value: v })}
                        />
                      );
                  }
                  case 'random': {
                      const strVal = String(input?.value ?? input?.default ?? '-1');
                      return (
                        <RandomInput
                          startLogo='🎲'
                          key={inputId}
                          label={input?.label}
                          description={input?.description}
                          value={strVal}
                          min={input?.min}
                          max={input?.max}
                          step={input?.step}
                          onChange={(v) => updateModelInput(clipId, inputId, { value: v })}
                        />
                      );
                  }

                  case 'image': {
                    const parseImageValue = (v: any) => {
                      if (!v) return null;
                      if (typeof v === 'object' && v !== null && (v.kind === 'media' || v.kind === 'clip')) return v;
                      if (typeof v === 'string') {
                        try {
                          const obj = JSON.parse(v);
                          if (obj && (obj.kind === 'media' || obj.kind === 'clip')) return obj;
                        } catch {}
                        return { kind: 'media', assetUrl: v } as any;
                      }
                      return null;
                    };
                    const currentVal: any = parseImageValue(input?.value);
                    return (
                      <ImageInput
                        clipId={clipId}
                        key={inputId}
                        label={input?.label}
                        description={input?.description}
                        value={currentVal}
                        panelSize={panelSize - 64}
                        onChange={(v: any) => updateModelInput(clipId, inputId, { value: v ? JSON.stringify(v) : '' })}
                      />
                    );
                  }
                  default:
                    return null;
                }
              })}
            </div>
          )
        })}
      </div>
      )}
    </div>
  )
}