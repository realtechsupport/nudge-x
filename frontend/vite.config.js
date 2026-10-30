import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// git-pages rewrites must include the deployment path on both sides.
function pagesRoutes() {
  let base;
  return {
    name: 'pages-routes',
    configResolved(config) {
      base = config.base;
    },
    generateBundle() {
      if (!base.startsWith('/') || base.startsWith('//')) {
        this.error('Use an absolute pathname for --base, for example / or /nudge-x/.');
      }
      this.emitFile({
        type: 'asset',
        fileName: '_redirects',
        source: `${base}site/* ${base}index.html 200\n`,
      });
    },
  };
}

export default defineConfig({
  plugins: [react(), pagesRoutes()],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          three: ['three'],
          react: ['react', 'react-dom'],
        },
      },
    },
  },
});
