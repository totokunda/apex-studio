import React, { useMemo } from "react";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { LuChevronDown, LuInfo } from "react-icons/lu";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface SelectOption {
  name: string;
  value: any;
}

interface SelectInputProps {
  label?: string;
  description?: string;
  value: string;
  onChange: (value: string) => void;
  defaultOption?: string;
  options: SelectOption[];
  useDropdown?: boolean;
}

const SelectInput: React.FC<SelectInputProps> = ({
  label,
  description,
  value,
  onChange,
  defaultOption,
  options,
  useDropdown: useDropdownProp,
}) => {
  const normalizedOptions = options;
  const useDropdown = useDropdownProp || normalizedOptions.length >= 4;
  const hasDescription = Boolean(description);

  const selectedOption = useMemo(
    () =>
      normalizedOptions.find((opt) => String(opt.value) === String(value)) ||
      normalizedOptions.find(
        (opt) => String(opt.value) === String(defaultOption),
      ),
    [normalizedOptions, value, defaultOption],
  );

  if (useDropdown) {
    if (hasDescription) {
      return (
        <div className="flex flex-col items-start w-full min-w-0 relative mt-1">
          <div className="flex items-center gap-1.5 mb-1.5">
            <label className="text-brand-light text-[10.5px] text-start font-medium">
              {label}
            </label>
            {description && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    className="text-brand-light/70 hover:text-brand-light focus:outline-none"
                  >
                    <LuInfo className="w-3 h-3" />
                  </button>
                </TooltipTrigger>
                <TooltipContent
                  sideOffset={6}
                  className="max-w-xs whitespace-pre-wrap text-[10px] font-poppins bg-brand-background border border-brand-light/10"
                >
                  {description}
                </TooltipContent>
              </Tooltip>
            )}
          </div>
          <div className="w-full">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="w-full py-1.5 px-3.5 rounded-[5px] font-medium bg-brand border border-brand-light/5 text-brand-light text-[10.5px] focus:outline-none focus:ring-2 focus:ring-brand-light/20 flex items-center justify-between hover:bg-brand-light/5 transition-all duration-200">
                  <span>{selectedOption?.name || "Select option"}</span>
                  <LuChevronDown className="w-3.5 h-3.5 text-brand-light/60" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-[var(--radix-dropdown-menu-trigger-width)] font-poppins bg-brand-background border-brand-light/10">
                <DropdownMenuRadioGroup
                  value={String(selectedOption?.value)}
                  onValueChange={onChange}
                >
                  {normalizedOptions.map((option) => (
                    <DropdownMenuRadioItem
                      key={option.value}
                      value={String(option.value)}
                      className="text-brand-light text-[10.5px] focus:bg-brand-light/10 font-medium focus:text-brand-lighter cursor-pointer"
                    >
                      {option.name}
                    </DropdownMenuRadioItem>
                  ))}
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      );
    }

    return (
      <div className="flex flex-row items-center w-full gap-x-2 min-w-0 relative mt-1">
        <label
          className={cn(
            "text-brand-light text-[10.5px] w-1/4 text-start font-medium",
            {
              "w-0": !label,
              "w-1/4": label,
            },
          )}
        >
          {label}
        </label>
        <div className="flex-1">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="w-full py-1.5 px-3.5 rounded-[5px] font-medium bg-brand border border-brand-light/5 text-brand-light text-[10.5px] focus:outline-none focus:ring-2 focus:ring-brand-light/20 flex items-center justify-between hover:bg-brand-light/5 transition-all duration-200">
                <span>{selectedOption?.name || "Select option"}</span>
                <LuChevronDown className="w-3.5 h-3.5 text-brand-light/60" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-[var(--radix-dropdown-menu-trigger-width)] font-poppins bg-brand-background border-brand-light/10">
              <DropdownMenuRadioGroup
                value={String(selectedOption?.value)}
                onValueChange={onChange}
              >
                {normalizedOptions.map((option) => (
                  <DropdownMenuRadioItem
                    key={option.value}
                    value={String(option.value)}
                    className="text-brand-light text-[10.5px] focus:bg-brand-light/10 font-medium focus:text-brand-lighter cursor-pointer"
                  >
                    {option.name}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    );
  }

  if (hasDescription) {
    return (
      <div className="flex flex-col items-start w-full min-w-0 relative mt-1">
        <div className="flex items-center gap-1.5 mb-1.5">
          <label className="text-brand-light text-[10.5px] text-start font-medium">
            {label}
          </label>
          {description && (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="text-brand-light/70 hover:text-brand-light focus:outline-none"
                >
                  <LuInfo className="w-3 h-3" />
                </button>
              </TooltipTrigger>
              <TooltipContent
                sideOffset={6}
                className="max-w-xs whitespace-pre-wrap text-[10px] font-poppins bg-brand-background border border-brand-light/10"
              >
                {description}
              </TooltipContent>
            </Tooltip>
          )}
        </div>
        <div className="flex flex-row w-full  rounded-[6px] overflow-hidden border border-brand-light/5 divide-x divide-brand-light/5">
          {normalizedOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => onChange(String(option.value))}
              className={cn(
                "flex-1 py-1.5 px-3 text-[10px] transition-all duration-200 font-medium",
                String(selectedOption?.value) === String(option.value)
                  ? "bg-brand-light/[0.1] text-brand-lighter"
                  : "bg-brand text-brand-light/60 hover:bg-brand-light/5",
              )}
            >
              {option.name}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-row items-center w-full gap-x-2 min-w-0 relative mt-1">
      <label
        className={cn(
          "text-brand-light text-[10.5px] w-1/4 text-start font-medium",
          {
            "w-0": !label,
            "w-1/4": label,
          },
        )}
      >
        {label}
      </label>
      <div className="flex flex-row flex-1  rounded-[6px] overflow-hidden ">
        {normalizedOptions.map((option) => (
          <button
            key={option.value}
            onClick={() => onChange(String(option.value))}
            className={cn(
              "flex-1 py-1.5 px-3 text-[10px]  font-medium",
              String(selectedOption?.value) === String(option.value)
                ? "bg-brand-light/10 text-brand-lighter"
                : "bg-brand text-brand-light/60 hover:bg-brand-light/5",
            )}
          >
            {option.name}
          </button>
        ))}
      </div>
    </div>
  );
};

export default SelectInput;
