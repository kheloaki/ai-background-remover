/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'export', // Export as static HTML for Flask to serve
};

export default nextConfig;