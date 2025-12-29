import { QueryClient } from "@tanstack/react-query";

declare global {
  // eslint-disable-next-line no-var
  var __APEX_QUERY_CLIENT__: QueryClient | undefined;
}

// Ensure a single QueryClient instance across imports/HMR.
export const queryClient: QueryClient =
  globalThis.__APEX_QUERY_CLIENT__ ?? (globalThis.__APEX_QUERY_CLIENT__ = new QueryClient());


