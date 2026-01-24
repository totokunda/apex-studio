import * as React from "react";
import { Check, ChevronDown, ChevronUp, Download, Star } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

export interface FontItem {
  value: string;
  label: string;
  isPremium?: boolean;
  isDownloaded?: boolean;
}

interface ComboboxProps {
  value?: string;
  onValueChange?: (value: string) => void;
  fonts: FontItem[];
  placeholder?: string;
}

export function Combobox({
  value,
  onValueChange,
  fonts,
  placeholder = "Select font...",
}: ComboboxProps) {
  const [open, setOpen] = React.useState(false);
  const [favorites, setFavorites] = React.useState<Set<string>>(new Set());

  const selectedFont = fonts.find((font) => font.value === value);

  const toggleFavorite = (fontValue: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setFavorites((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(fontValue)) {
        newSet.delete(fontValue);
      } else {
        newSet.add(fontValue);
      }
      return newSet;
    });
  };

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className="w-full justify-between rounded bg-brand dark border-brand-light/5 pr-2 pl-2.5 text-brand-light hover:border-brand-light/10 text-xs h-7"
        >
          <span
            className="text-left font-normal"
            style={{ fontFamily: selectedFont?.value || "inherit" }}
          >
            {selectedFont ? selectedFont.label : placeholder}
          </span>
          <div className="flex flex-col items-center">
            <ChevronUp className="h-2.5! w-2.5! shrink-0 opacity-50" />
            <ChevronDown className="h-2.5! w-2.5! shrink-0 opacity-50" />
          </div>
        </Button>
      </PopoverTrigger>
      <PopoverContent
        className="w-[300px] p-0 bg-brand-background-dark border-brand-light/10"
        align="start"
      >
        <Command className="bg-brand-background-dark">
          <CommandInput
            placeholder="Search for text"
            className="text-brand-light placeholder:text-brand-light/40 text-xs"
          />
          <CommandList>
            <CommandEmpty className="text-brand-light/40 p-4 text-xs">
              No font found.
            </CommandEmpty>
            <CommandGroup heading="Presets" className="px-2 py-2">
              <CommandItem
                value="system"
                onSelect={() => {
                  onValueChange?.("system");
                  setOpen(false);
                }}
                className="flex items-center justify-between h-8 py-1.5 px-3 hover:bg-brand-light/5 data-[selected=true]:bg-brand-light/5 rounded-sm"
              >
                <div className="flex items-center gap-2">
                  {value === "system" && (
                    <Check className="h-4 w-4 text-brand-accent" />
                  )}
                  {value !== "system" && <span className="w-4" />}
                  <span className="text-brand-light text-xs">System</span>
                </div>
                <button
                  onClick={(e) => toggleFavorite("system", e)}
                  className="text-brand-light/40 hover:text-brand-accent transition-colors"
                >
                  <Star
                    className={cn(
                      "h-4 w-4",
                      favorites.has("system") &&
                        "fill-brand-accent text-brand-accent",
                    )}
                  />
                </button>
              </CommandItem>
            </CommandGroup>
            <CommandGroup className="px-2 pb-2">
              {fonts.map((font) => (
                <CommandItem
                  key={font.value}
                  value={font.value}
                  onSelect={(currentValue) => {
                    onValueChange?.(currentValue === value ? "" : currentValue);
                    setOpen(false);
                  }}
                  className="flex items-center justify-between py-1 px-3 h-8 hover:bg-brand-light/5 data-[selected=true]:bg-brand-light/5 rounded-sm group"
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    {value === font.value && (
                      <Check className="h-4 w-4 text-brand-accent shrink-0" />
                    )}
                    {value !== font.value && <span className="w-4 shrink-0" />}
                    <span
                      className="text-brand-light text-xs truncate"
                      style={{ fontFamily: font.value }}
                    >
                      {font.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-1 shrink-0 opacity-0 group-hover:opacity-100 transition-opacity">
                    {!font.isDownloaded && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          // Handle download
                        }}
                        className="text-brand-light/40 hover:text-brand-accent transition-colors p-1"
                      >
                        <Download className="h-4 w-4" />
                      </button>
                    )}
                    <button
                      onClick={(e) => toggleFavorite(font.value, e)}
                      className="text-brand-light/40 hover:text-brand-accent transition-colors p-1"
                    >
                      <Star
                        className={cn(
                          "h-4 w-4",
                          favorites.has(font.value) &&
                            "fill-brand-accent text-brand-accent",
                        )}
                      />
                    </button>
                  </div>
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
