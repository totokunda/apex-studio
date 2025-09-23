import { HiFilm } from "react-icons/hi";
import { LuTrainTrack } from "react-icons/lu";
import { HiOutlineTemplate } from "react-icons/hi";
import { HiOutlineLightBulb } from "react-icons/hi";
import { HiOutlineSparkles } from "react-icons/hi";
import { MdOutlineMovieFilter } from "react-icons/md";

import SidebarTrigger from '../media/MediaModelTrigger'

interface LeftSidebarProps {
  onOpen: () => void;
  onClose: () => void;
}

const LeftSidebar:React.FC<LeftSidebarProps> = ({ onOpen, onClose }) => {
  return (
    <div className="h-full w-20 bg-brand-background-dark flex flex-col gap-y-2.5 pt-20 border-r border-brand-light/5 px-2">
      <SidebarTrigger icon={<HiFilm className="h-4 w-4" />} title="Media" section="media" onOpen={onOpen} onClose={onClose}  />
      <SidebarTrigger icon={<HiOutlineLightBulb className="h-4 w-4 stroke-2" />} title="Models" section="models" onOpen={onOpen} onClose={onClose} />
      <SidebarTrigger icon={<LuTrainTrack className="h-4 w-4 stroke-2" />} title="Tracks" section="tracks" onOpen={onOpen} onClose={onClose} />
      <SidebarTrigger icon={<MdOutlineMovieFilter className="h-4 w-4 " />} title="LoRAs" section="loras" onOpen={onOpen} onClose={onClose} />
      <SidebarTrigger icon={<HiOutlineTemplate className="h-4 w-4 stroke-2" />} title="Templates" section="templates" onOpen={onOpen} onClose={onClose} />
    </div>
  )
}

export default LeftSidebar