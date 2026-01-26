import React, { useEffect, useRef, useState } from "react";

interface TextInputProps {
  label?: string;
  description?: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  defaultValue?: string;
}

const TextInput: React.FC<TextInputProps> = ({
  label,
  description,
  value,
  onChange,
  placeholder,
  defaultValue,
}) => {
  const [internalValue, setInternalValue] = useState(
    value ?? defaultValue ?? "",
  );
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const debounceTimeoutRef = useRef<number | null>(null);

  // Keep internal value in sync if parent-controlled value changes externally
  useEffect(() => {
    setInternalValue(value);
  }, [value]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height =
        textareaRef.current.scrollHeight + "px";
    }
  }, [internalValue]);

  // Clear any pending debounce on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current !== null) {
        window.clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    setInternalValue(newValue);

    if (debounceTimeoutRef.current !== null) {
      window.clearTimeout(debounceTimeoutRef.current);
    }

    debounceTimeoutRef.current = window.setTimeout(() => {
      onChange(newValue);
    }, 300);
  };

  return (
    <div className="flex flex-col items-start w-full gap-y-1 min-w-0 relative h-full px-3  py-4 pt-8 placeholder:text-brand-light/40  text-brand-light text-[11px] font-normal border border-brand-light/5 shadow bg-brand rounded-[7px]">
      <label className="text-brand-light text-[10.5px] w-full text-start font-medium absolute top-3 left-3">
        {label}
      </label>
      
      <textarea
        ref={textareaRef}
        defaultValue={defaultValue}
        value={internalValue}
        onChange={handleChange}
        placeholder={placeholder}
        rows={4}
        className=" w-full h-full resize-none overflow-hidden dark focus-visible:outline-none focus-visible:ring-0 pb-1"
      />
      {description && (
        <span className="text-brand-light/80 text-[9.5px] text-start  w-fit mr-2">
          {description}
        </span>
      )}
    </div>
  );
};

export default TextInput;
