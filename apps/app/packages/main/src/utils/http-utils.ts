

export function getRange(
    rangeHeader: string | null | undefined,
    size: number
  ): { start: number; end: number } | { unsatisfiable: true } | null {
    if (!rangeHeader) return null;
    if (!Number.isFinite(size) || size < 0) return null;
  
    const h = rangeHeader.trim();
  
    // We only support single ranges (no commas)
    if (h.includes(",")) return null;
  
    const m = /^bytes=(\d*)-(\d*)$/i.exec(h);
    if (!m) return null;
  
    const startStr = m[1];
    const endStr = m[2];
  
    // "bytes=-" is invalid
    if (startStr === "" && endStr === "") return { unsatisfiable: true };
  
    // Empty file: any range is unsatisfiable
    if (size === 0) return { unsatisfiable: true };
  
    let start: number;
    let end: number;
  
    if (startStr === "") {
      // Suffix range: "bytes=-N" => last N bytes
      const suffixLen = Number(endStr);
      if (!Number.isFinite(suffixLen) || suffixLen <= 0) return { unsatisfiable: true };
  
      start = Math.max(0, size - suffixLen);
      end = size - 1;
    } else {
      start = Number(startStr);
      if (!Number.isFinite(start) || start < 0) return { unsatisfiable: true };
  
      if (endStr === "") {
        // Open-ended: "bytes=N-" => N..EOF
        end = size - 1;
      } else {
        end = Number(endStr);
        if (!Number.isFinite(end) || end < 0) return { unsatisfiable: true };
      }
    }
  
    // Clamp end to EOF
    end = Math.min(end, size - 1);
  
    // Satisfiability checks
    if (start >= size) return { unsatisfiable: true };
    if (start > end) return { unsatisfiable: true };
  
    return { start, end };
  }