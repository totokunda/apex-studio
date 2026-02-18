import React, { useCallback } from "react";
import { useSidebarStore } from "@/lib/sidebar";
import { SidebarSection } from "@/lib/types";
import { cn } from "@/lib/utils";

interface MediaModelTriggerProps {
  onClick?: () => void;
  onPointerEnter?: () => void;
  icon: React.ReactNode;
  title: string;
  section: SidebarSection;
  onOpen?: () => void;
  onClose?: () => void;
}

const MediaModelTrigger: React.FC<MediaModelTriggerProps> = ({
  onClick,
  onPointerEnter,
  icon,
  title,
  section,
  onOpen,
}) => {
  const {
    openSection,
    section: currentSection,
  } = useSidebarStore();
  const handleClick = useCallback(() => {
    onClick?.();
    openSection(section, currentSection === null ? onOpen : undefined);
  }, [section, currentSection, openSection, onClick, onOpen]);

  return (
    <button
      type="button"
      data-sidebar-section={section}
      aria-pressed={currentSection === section}
      aria-label={title}
      onClick={handleClick}
      onPointerEnter={onPointerEnter}
      className={cn(
        "flex flex-row font-medium items-center px-3 border-0 justify-center border-transparent py-1.5 gap-x-1.5 rounded text-brand-light/60 transition-all duration-200 cursor-pointer whitespace-nowrap focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-accent-shade/60",
        {
          "text-brand-light bg-brand-background-light": currentSection === section,
          "border-brand-accent-shade/40": currentSection === section,
          "hover:border-brand-light/5": currentSection !== section,
          "hover:bg-brand-light/5": currentSection !== section,
          "hover:text-brand-lighter": currentSection !== section,
        },
      )}
    >
      {icon}
      <span className="text-[11px]">{title}</span>
    </button>
  );
};

export default MediaModelTrigger;
