import React, { useEffect, useRef } from 'react'

interface TextInputProps {
  label?: string;
  description?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  defaultValue?: string;
}

const TextInput: React.FC<TextInputProps> = ({ label, description, value, onChange, placeholder, defaultValue }) => {
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = textareaRef.current.scrollHeight + 'px';
    }
  }, [value]);

  return (
    <div className="flex flex-col items-start w-full gap-y-1 min-w-0 relative h-full px-3 py-8 placeholder:text-brand-light/40  text-brand-light text-[11px] font-normal border border-brand-light/5 shadow bg-brand rounded-[7px]">
      <label className="text-brand-light text-[10.5px] w-full text-start font-medium absolute top-3 left-3">{label}</label>
      {description && <span className="text-brand-light/80 text-[9.5px] w-full text-start mb-0.5 absolute bottom-3 left-3">{description}</span>}
      <textarea
        ref={textareaRef}
        defaultValue={value === undefined || value === '' ? defaultValue : value}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={4}
        className=" w-full h-full resize-none overflow-hidden dark focus-visible:outline-none focus-visible:ring-0"
      />
    </div>
  )
}

export default TextInput