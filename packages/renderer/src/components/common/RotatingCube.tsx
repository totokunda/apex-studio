import React, { useEffect, useMemo, useRef } from "react";
import Konva from "konva";
import { Group, Line } from "react-konva";

type Vec3 = { x: number; y: number; z: number };

// Persist rotation phase across unmounts; reset only when phaseKey changes
let GLOBAL_CUBE_EPOCH_MS = Date.now();

type RotatingCubeProps = {
  size?: number; // cube edge length in px (projected scale)
  x?: number; // center x within parent Group
  y?: number; // center y within parent Group
  opacity?: number;
  stroke?: string;
  strokeWidth?: number;
  listening?: boolean;
  baseColors?: string[]; // optional, 6 face base colors
  phaseKey?: string; // optional, reset animation phase when this changes
};

const RotatingCube: React.FC<RotatingCubeProps> = ({
  size = 18,
  x = 0,
  y = 0,
  opacity = 0.95,
  stroke = "#ffffff",
  strokeWidth = 0.75,
  listening = false,
  baseColors,
  phaseKey,
}) => {
  const groupRef = useRef<Konva.Group>(null);
  const faceRefs = useRef<Array<Konva.Line | null>>([]);
  const startEpochRef = useRef<number>(GLOBAL_CUBE_EPOCH_MS);

  const verts = useMemo<Vec3[]>(() => {
    const s = size / 2;
    return [
      { x: -s, y: -s, z: -s },
      { x: s, y: -s, z: -s },
      { x: s, y: s, z: -s },
      { x: -s, y: s, z: -s },
      { x: -s, y: -s, z: s },
      { x: s, y: -s, z: s },
      { x: s, y: s, z: s },
      { x: -s, y: s, z: s },
    ];
  }, [size]);

  const faces = useMemo<number[][]>(
    () => [
      [0, 1, 2, 3],
      [4, 5, 6, 7],
      [0, 1, 5, 4],
      [2, 3, 7, 6],
      [1, 2, 6, 5],
      [0, 3, 7, 4],
    ],
    [],
  );

  const faceBaseColors = useMemo(() => {
    const fallback = [
      "#4f83ff",
      "#6ee7b7",
      "#f87171",
      "#fbbf24",
      "#a78bfa",
      "#60a5fa",
    ];
    if (!baseColors || baseColors.length < 6) return fallback;
    return baseColors.slice(0, 6);
  }, [baseColors]);

  const rotate = (v: Vec3, ax: number, ay: number, az: number): Vec3 => {
    let x1 = v.x,
      y1 = v.y * Math.cos(ax) - v.z * Math.sin(ax),
      z1 = v.y * Math.sin(ax) + v.z * Math.cos(ax);
    const x2 = x1 * Math.cos(ay) + z1 * Math.sin(ay);
    const z2 = -x1 * Math.sin(ay) + z1 * Math.cos(ay);
    x1 = x2;
    z1 = z2;
    const x3 = x1 * Math.cos(az) - y1 * Math.sin(az);
    const y3 = x1 * Math.sin(az) + y1 * Math.cos(az);
    return { x: x3, y: y3, z: z1 };
  };

  // Use orthographic projection so the cube maintains perfect proportions on screen
  const project = (v: Vec3) => {
    return { x: v.x, y: v.y, z: v.z };
  };

  const faceNormal = (a: Vec3, b: Vec3, c: Vec3): Vec3 => {
    const ab = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
    const ac = { x: c.x - a.x, y: c.y - a.y, z: c.z - a.z };
    return {
      x: ab.y * ac.z - ab.z * ac.y,
      y: ab.z * ac.x - ab.x * ac.z,
      z: ab.x * ac.y - ab.y * ac.x,
    };
  };

  const shadeHex = (hex: string, factor: number) => {
    const c = hex.replace("#", "");
    const r = parseInt(c.slice(0, 2), 16);
    const g = parseInt(c.slice(2, 4), 16);
    const b = parseInt(c.slice(4, 6), 16);
    const mix = (v: number) =>
      Math.round(v + ((255 - v) * (factor - 0.3)) / 0.7);
    const toHex = (v: number) => v.toString(16).padStart(2, "0");
    return `#${toHex(mix(r))}${toHex(mix(g))}${toHex(mix(b))}`;
  };

  // Reset phase only when explicit phaseKey changes
  useEffect(() => {
    if (phaseKey !== undefined) {
      const now = Date.now();
      GLOBAL_CUBE_EPOCH_MS = now;
      startEpochRef.current = now;
    }
  }, [phaseKey]);

  useEffect(() => {
    const light = { x: 0.4, y: -0.6, z: 1.0 };
    const anim = new Konva.Animation(() => {
      const tSec = (Date.now() - startEpochRef.current) / 1000;
      // Angular speeds (rad/sec) chosen to be incommensurate for a pleasing loop
      const ax = tSec * 1.2;
      const ay = tSec * 1.62;
      const az = tSec * 1.02;
      const rv = verts.map((v) => rotate(v, ax, ay, az));
      const faceData = faces.map((idxs, i) => {
        const a = rv[idxs[0]],
          b = rv[idxs[1]],
          c = rv[idxs[2]],
          d = rv[idxs[3]];
        const n0 = faceNormal(a, b, c);
        const nLen = Math.hypot(n0.x, n0.y, n0.z) || 1;
        const n = { x: n0.x / nLen, y: n0.y / nLen, z: n0.z / nLen };
        const lLen = Math.hypot(light.x, light.y, light.z) || 1;
        const l = { x: light.x / lLen, y: light.y / lLen, z: light.z / lLen };
        const ndotl = Math.max(0, n.x * l.x + n.y * l.y + n.z * l.z);
        const brightness = 0.3 + 0.7 * ndotl;
        const pv = idxs.map((j) => project(rv[j]));
        const points = pv.flatMap((p) => [p.x, p.y]);
        const depth = (a.z + b.z + c.z + d.z) / 4;
        return {
          i,
          points,
          depth,
          fill: shadeHex(faceBaseColors[i], brightness),
        };
      });
      faceData.sort((a, b) => a.depth - b.depth);
      faceData.forEach((fd, order) => {
        const node = faceRefs.current[order];
        if (!node) return;
        node.points(fd.points);
        node.fill(fd.fill);
      });
    }, groupRef.current?.getLayer());
    anim.start();
    return () => {
      anim.stop();
    };
  }, [verts, faces, faceBaseColors]);

  return (
    <Group ref={groupRef} x={x} y={y} opacity={opacity} listening={listening}>
      {faces.map((_, idx) => (
        <Line
          key={idx}
          ref={(n) => {
            faceRefs.current[idx] = n;
          }}
          closed
          stroke={stroke}
          strokeWidth={strokeWidth}
          lineJoin="round"
          lineCap="round"
          miterLimit={1}
          strokeScaleEnabled={false}
          listening={false}
        />
      ))}
    </Group>
  );
};

export default RotatingCube;
