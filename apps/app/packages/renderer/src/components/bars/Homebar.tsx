import { LuMenu } from "react-icons/lu";

const Homebar = () => {
  return (
    <div className="fixed top-3.5 left-7 z-50 gap-x-2 cursor-pointer pb-6 border-b border-brand flex items-center justify-between">
      <div>
        <LuMenu className="w-[22px] h-[22px] text-brand-light" />
      </div>
      <div>
        <span className="text-brand-light font-semibold text-sm">
          Apex Studio
        </span>
      </div>
    </div>
  );
};

export default Homebar;
