import React, { useCallback } from "react";
import { useSidebarStore } from "@/lib/sidebar";
import { SidebarSection } from "@/lib/types";
import { cn } from "@/lib/utils";

interface MediaModelTriggerProps {
  onClick?: () => void;
  icon: React.ReactNode;
  title: string;
  section: SidebarSection;
  onOpen?: () => void;
  onClose?: () => void;
}

const MediaModelTrigger: React.FC<MediaModelTriggerProps> = ({
  onClick,
  icon,
  title,
  section,
  onOpen,
}) => {
  const {
    openSection,
    closeSection,
    section: currentSection,
  } = useSidebarStore();
  const handleClick = useCallback(() => {
    onClick?.();
    openSection(section, currentSection === null ? onOpen : undefined);
  }, [section, currentSection, closeSection, openSection]);

  return (
    <div
      onClick={handleClick}
      className={cn(
        "flex flex-row font-medium items-center justify-center border border-transparent px-2.5 py-1.5 gap-x-1.5 rounded  text-brand-light/60  transition-all duration-200 cursor-pointer",
        {
          "bg-brand-accent": currentSection === section,
          "text-brand-lighter": currentSection === section,
          "border-brand-accent": currentSection === section,
          "hover:border-brand-light/5": currentSection !== section,
          "hover:bg-brand-light/5": currentSection !== section,
          "hover:text-brand-light": currentSection !== section,
        },
      )}
    >
      {icon}
      <span className="text-[11px]">{title}</span>
    </div>
  );
};

export default MediaModelTrigger;
