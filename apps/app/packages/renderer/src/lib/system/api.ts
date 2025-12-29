import { getSystemMemory } from "@app/preload";

export type MemoryBreakdown = {
  total: number;
  used: number;
  available?: number;
  percent: number;
};

export type GpuAdapter = {
  index: number;
  total: number;
  free: number;
  used: number;
  percent: number;
};

export type SystemMemoryResponse = {
  unified: MemoryBreakdown | null;
  cpu: MemoryBreakdown | null;
  gpu: {
    device_type: "cuda";
    count: number;
    adapters: GpuAdapter[];
    total: number;
    used: number;
    percent: number;
  } | null;
  device_type: "mps" | "cuda" | "cpu";
};

export async function fetchSystemMemory(): Promise<SystemMemoryResponse | null> {
  const res = await getSystemMemory();
  if (!res.success) return null;
  return res.data as SystemMemoryResponse;
}
