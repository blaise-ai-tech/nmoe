import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  // DuckDB is a native Node addon; keep it external to the server bundle.
  // NVIZ only uses it in server routes.
  serverExternalPackages: ['@duckdb/node-api'],
};

export default nextConfig;
