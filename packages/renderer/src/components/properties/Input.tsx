import { cn } from '@/lib/utils';
import React, { useEffect, useState } from 'react'
import { LuChevronDown, LuChevronUp } from 'react-icons/lu';

interface InputProps<T> {
  className?: string;
  label?: string;
  value: T;
  onChange: (value: T) => void;
  startLogo?: string;
  canStep?: boolean;
  step?: number;
  max?: number;
  min?: number;
  emptyLabel?: boolean;
}

const Input: React.FC<InputProps<string>> = ({ value, onChange, label, className, startLogo, canStep, step, max, min, emptyLabel }) => {
    const [tempValue, setTempValue] = useState(value);
    const lastValueRef = React.useRef(value);
    
    useEffect(() => {
        setTempValue(value);
        lastValueRef.current = value;
    }, [value]);

    const handleBlur = () => {
        if (tempValue === value) return; // No change, skip
        const oldValue = lastValueRef.current;
        onChange(tempValue);
        
        // Check if change was accepted by seeing if value prop changed
        requestAnimationFrame(() => {
            // Schedule another check to see if parent updated the value
            requestAnimationFrame(() => {
                if (lastValueRef.current === oldValue) {
                    // Value didn't change, reset to current value
                    setTempValue(lastValueRef.current);
                }
            });
        });
    };

    const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            if (tempValue !== value) {
                const oldValue = lastValueRef.current;
                onChange(tempValue);
                requestAnimationFrame(() => {
                    requestAnimationFrame(() => {
                        if (lastValueRef.current === oldValue) {
                            setTempValue(lastValueRef.current);
                        }
                    });
                });
            }
            e.currentTarget.blur();
        } else if (e.key === 'Escape') {
            setTempValue(value);
            e.currentTarget.blur();
        }
    };
    
  return (
    <div className="flex flex-col items-start w-full gap-y-1">
    <label className="text-brand-light  text-[11px] w-full text-start mb-0.5">{label}</label>
    {emptyLabel && <span className="mb-2.5"></span>}

    <div className="relative w-full flex flex-row items-center">
            {startLogo && <span className="text-brand-light/50 font-medium text-[11px] absolute left-2 top-1/2 -translate-y-1/2">{startLogo}</span>}
            <input className={cn(`w-full h-6 px-1.5 text-brand-light text-[11px] font-normal items-center border border-brand-light/10 p-1  bg-brand ${className}`, {
                'pl-6': startLogo,
                "rounded-l": canStep,
                "rounded": !canStep,
            })} 
            value={tempValue} 
            onChange={(e) => setTempValue(e.target.value)}
            onBlur={handleBlur}
            onKeyDown={handleKeyDown} />
        {canStep && <div className="flex flex-col items-center w-6 justify-center divide-y divide-brand-light/10 bg-brand  h-6 cursor-pointer rounded-r">
            <button className="w-full h-full px-1 hover:bg-brand-light/10 transition-all duration-200 flex items-center justify-center" 
            onClick={() => {
              const numValue = Number(value);
              if (isNaN(numValue) || !isFinite(numValue)) return;
              const changedValue = Math.min(numValue + (step ?? 0), max ?? numValue + (step ?? 0));
              onChange(changedValue.toString());
            }}>
            <LuChevronUp className="w-2 h-2 cursor-pointer text-brand-light" />
          </button>
          <button className="w-full h-full px-1 hover:bg-brand-light/10 transition-all duration-200 flex items-center justify-center" 
          onClick={() => {
            const numValue = Number(value);
            if (isNaN(numValue) || !isFinite(numValue)) return;
            const changedValue = Math.max(numValue - (step ?? 0), min ?? numValue - (step ?? 0));
            onChange(changedValue.toString());
          }}>
          <LuChevronDown className="w-2 h-2 cursor-pointer text-brand-light" />
          </button>
        </div>
        }
    </div>
    </div>
  )
}

export default Input