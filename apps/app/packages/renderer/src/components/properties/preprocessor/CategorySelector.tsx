import React from "react";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { LuChevronDown } from "react-icons/lu";

interface CategoryOption {
  name: string;
  value: any;
}

interface CategorySelectorProps {
  value: string;
  onChange: (value: string) => void;
  options: CategoryOption[];
}

const CategorySelector: React.FC<CategorySelectorProps> = ({
  value,
  onChange,
  options,
}) => {
  const normalizedOptions = options;
  const useDropdown = normalizedOptions.length >= 4;

  const selectedOption = normalizedOptions.find(
    (opt) => String(opt.value) === String(value),
  );

  if (useDropdown) {
    return (
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="w-full py-2 px-3 rounded-[6px] bg-brand-background-light border font-medium border-brand-light/10 text-brand-light text-[11px] focus:outline-none focus:ring-2 focus:ring-brand-light/20 flex items-center justify-between hover:bg-brand-light/5 transition-all duration-200">
            <span>{selectedOption?.name || "Select option"}</span>
            <LuChevronDown className="w-3.5 h-3.5 text-brand-light/60" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-[var(--radix-dropdown-menu-trigger-width)] font-poppins bg-brand border-brand-light/10">
          <DropdownMenuRadioGroup
            value={String(value)}
            onValueChange={onChange}
          >
            {normalizedOptions.map((option) => (
              <DropdownMenuRadioItem
                key={option.value}
                value={String(option.value)}
                className="text-brand-light text-[10.5px] font-medium focus:bg-brand-light/10 focus:text-brand-lighter cursor-pointer"
              >
                {option.name}
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
        </DropdownMenuContent>
      </DropdownMenu>
    );
  }

  return (
    <div className="flex flex-row w-full border border-brand-light/10 rounded-[6px] overflow-hidden">
      {normalizedOptions.map((option, index) => (
        <button
          key={option.value}
          onClick={() => onChange(String(option.value))}
          className={cn(
            "flex-1 py-1.5 px-3 text-[10.5px] font-medium transition-all duration-200",
            String(value) === String(option.value)
              ? "bg-brand-background-light text-brand-lighter"
              : "bg-brand text-brand-light/60 hover:bg-brand-light/5",
            index !== 0 && "border-l border-brand-light/10",
          )}
        >
          {option.name}
        </button>
      ))}
    </div>
  );
};

export default CategorySelector;
