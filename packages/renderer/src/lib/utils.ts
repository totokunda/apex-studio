import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import Konva from "konva";
import { useEffect, useRef, useState } from "react";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
