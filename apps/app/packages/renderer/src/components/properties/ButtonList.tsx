import React from "react";
import { cn } from "@/lib/utils";

interface ButtonListProps {
  buttons: {
    value: string;
    icon: React.ReactNode;
    onClick?: () => void;
  }[];
  selected: string;
  onSelect: (value: string) => void;
}

const ButtonList: React.FC<ButtonListProps> = ({
  buttons,
  selected,
  onSelect,
}) => {
  return (
    <div className="flex flex-row  bg-brand  divide-x divide-brand-light/10 rounded w-full">
      {buttons.map((button, index) => (
        <button
          key={index}
          onClick={() => {
            onSelect(button.value);
            button.onClick?.();
          }}
          className={cn(
            "flex items-center p-1.5 justify-center w-full text-brand-light/70 cursor-pointer hover:text-brand-light hover:bg-brand-light/10 duration-200 transition-all",
            index === 0 && "rounded-l",
            index === buttons.length - 1 && "rounded-r",
            selected === button.value && "bg-brand-light/10 text-brand-light",
          )}
        >
          {button.icon}
        </button>
      ))}
    </div>
  );
};

export default ButtonList;
