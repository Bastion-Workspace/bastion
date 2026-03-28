import { defineConfig, transformWithEsbuild } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'node:fs';
import fsPromises from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Serves the /pmtiles URL prefix from /data/pmtiles (Docker volume) with Range request support,
 * matching nginx behavior for MapLibre PMTiles.
 */
function pmtilesDevPlugin() {
  const root = '/data/pmtiles';
  return {
    name: 'pmtiles-range-static',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = req.url || '';
        if (!url.startsWith('/pmtiles/')) return next();
        const pathname = url.split('?')[0];
        const rel = decodeURIComponent(pathname.slice('/pmtiles/'.length));
        if (!rel || rel.includes('..')) return next();
        const resolvedRoot = path.resolve(root);
        const filePath = path.join(resolvedRoot, rel);
        if (!filePath.startsWith(resolvedRoot)) return next();

        fs.stat(filePath, (err, st) => {
          if (err || !st.isFile()) return next();
          const size = st.size;
          res.setHeader('Accept-Ranges', 'bytes');
          res.setHeader('Access-Control-Allow-Origin', '*');
          res.setHeader('Access-Control-Allow-Headers', 'Range');
          res.setHeader('Access-Control-Expose-Headers', 'Content-Range, Content-Length');
          res.setHeader('Cache-Control', 'public, max-age=86400');

          const range = req.headers.range;
          if (range) {
            const m = /^bytes=(\d*)-(\d*)$/.exec(range);
            if (!m) return next();
            let start = m[1] ? parseInt(m[1], 10) : 0;
            let end = m[2] ? parseInt(m[2], 10) : size - 1;
            if (Number.isNaN(start)) start = 0;
            if (Number.isNaN(end) || end >= size) end = size - 1;
            if (start > end) {
              res.statusCode = 416;
              return res.end();
            }
            res.statusCode = 206;
            res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
            res.setHeader('Content-Length', String(end - start + 1));
            fs.createReadStream(filePath, { start, end }).pipe(res);
          } else {
            res.setHeader('Content-Length', String(size));
            fs.createReadStream(filePath).pipe(res);
          }
        });
      });
    },
  };
}

/** Vite prefixes absolute fs paths (see packages/vite/src/node/constants.ts `FS_PREFIX`). */
const VITE_FS_PREFIX = '/@fs/';

function idToFsPath(id) {
  const pathOnly = id.split('?')[0];
  if (pathOnly.startsWith('file://')) {
    return fileURLToPath(pathOnly);
  }
  if (pathOnly.startsWith(VITE_FS_PREFIX)) {
    return pathOnly.slice(VITE_FS_PREFIX.length);
  }
  return pathOnly;
}

function isUnderSrc(fsPath) {
  const n = fsPath.replace(/\\/g, '/');
  return /(^|\/)src\//.test(n);
}

function resolveSrcJsFilePath(id) {
  let p = idToFsPath(id);
  p = p.replace(/\\/g, '/');
  if (p.startsWith('file://')) {
    p = fileURLToPath(p);
  }
  if (fs.existsSync(p) && path.isAbsolute(p)) {
    return p;
  }
  const cwd = process.cwd();
  const candidates = [
    p,
    path.resolve(cwd, p.replace(/^\//, '')),
    path.resolve(cwd, p),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return null;
}

// CRA-style JSX in src .js files (nested dirs). Module ids may use Vite /@fs/ plus absolute path; our earlier
// load hook read the wrong path, fell back to the default loader, and import-analysis saw raw JSX.
function jsxInSrcJsPlugin() {
  return {
    name: 'vite-jsx-in-src-js',
    enforce: 'pre',
    async load(id) {
      const pathOnly = id.split('?')[0];
      if (pathOnly.startsWith('\0')) return null;
      if (!pathOnly.endsWith('.js')) return null;
      if (pathOnly.includes('node_modules')) return null;
      const fsPath = resolveSrcJsFilePath(id);
      if (!fsPath || !isUnderSrc(fsPath)) return null;
      let code;
      try {
        code = await fsPromises.readFile(fsPath, 'utf-8');
      } catch {
        return null;
      }
      const result = await transformWithEsbuild(code, id, {
        loader: 'jsx',
        jsx: 'automatic',
      });
      return { code: result.code, map: result.map };
    },
  };
}

export default defineConfig({
  plugins: [
    jsxInSrcJsPlugin(),
    react({
      // CRA used .js for JSX; compile JSX in src/**/*.js without renaming files.
      include: /\/src\/.*\.(js|jsx)$/,
    }),
    pmtilesDevPlugin(),
  ],
  // CRA used .js for JSX. Vite's esbuild step defaults to ts/jsx/tsx only; setting `include`
  // replaces that list, so we must list those extensions again plus app src/*.js.
  esbuild: {
    jsx: 'automatic',
    include: [/\.(tsx?|jsx)$/, /\/src\/.*\.js$/],
  },
  optimizeDeps: {
    esbuildOptions: {
      loader: {
        '.js': 'jsx',
      },
    },
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    // Allow dev UI behind reverse proxy / TLS (Host: bastion.pilbeams.net, etc.).
    // Override with VITE_DEV_ALLOWED_HOSTS=host1,host2 for an explicit allowlist.
    allowedHosts: (() => {
      const raw = process.env.VITE_DEV_ALLOWED_HOSTS;
      if (!raw || !raw.trim()) return true;
      const list = raw.split(',').map((h) => h.trim()).filter(Boolean);
      return list.length ? list : true;
    })(),
    proxy: {
      '/api': { target: 'http://backend:8000', changeOrigin: true, ws: true },
      '/dav': {
        target: 'http://webdav:8001',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/dav/, '') || '/',
      },
      '/health': { target: 'http://backend:8000', changeOrigin: true },
      '/static/images': { target: 'http://backend:8000', changeOrigin: true },
    },
    watch: { usePolling: true },
  },
  build: {
    outDir: 'build',
  },
});
