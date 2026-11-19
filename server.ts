import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { fileURLToPath } from 'node:url';
import express from 'express';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT) || 3000;
const WORKER_SCRIPT = path.resolve(__dirname, 'python_engine', 'pycut_splitter', 'worker.py');
const SAMPLE_SLIDES_PATH = path.resolve(__dirname, 'public', 'examples', 'sample_slides.png');

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
  timer: NodeJS.Timeout;
}

class PythonWorkerPool {
  private proc: ChildProcessWithoutNullStreams | null = null;
  private readyPromise: Promise<void> | null = null;
  private pending = new Map<number, PendingRequest>();
  private nextId = 1;

  private ensureStarted(): Promise<void> {
    if (this.proc && this.readyPromise) {
      return this.readyPromise;
    }

    this.readyPromise = new Promise<void>((resolveReady, rejectReady) => {
      const venvPy = path.resolve(__dirname, '.venv', 'bin', 'python');
      const pyBin = fs.existsSync(venvPy)
        ? venvPy
        : fs.existsSync('/opt/py312/bin/python')
          ? '/opt/py312/bin/python'
          : 'python3.12';
      const child = spawn(pyBin, ['-u', WORKER_SCRIPT], {
        cwd: __dirname,
        env: { ...process.env, PYTHONUNBUFFERED: '1' },
        stdio: ['pipe', 'pipe', 'pipe'],
      });
      this.proc = child;
      let isReady = false;

      const rl = readline.createInterface({ input: child.stdout, crlfDelay: Infinity });
      rl.on('line', (line) => {
        if (!line.trim()) return;
        let msg: { ready?: boolean; id?: number; ok?: boolean; data?: unknown; error?: string };
        try {
          msg = JSON.parse(line);
        } catch {
          return;
        }
        if (!isReady && msg.ready) {
          isReady = true;
          resolveReady();
          return;
        }
        if (typeof msg.id === 'number') {
          const entry = this.pending.get(msg.id);
          if (!entry) return;
          this.pending.delete(msg.id);
          clearTimeout(entry.timer);
          if (msg.ok) {
            entry.resolve(msg.data);
          } else {
            entry.reject(new Error(msg.error || 'Python worker error'));
          }
        }
      });

      child.stderr.on('data', (chunk) => {
        const txt = String(chunk).trim();
        if (txt) {
          console.error('[pycut-worker stderr]', txt);
        }
      });

      child.on('exit', (code) => {
        this.proc = null;
        this.readyPromise = null;
        if (!isReady) {
          rejectReady(new Error(`Python worker exited prematurely with code ${code}`));
        }
        for (const [id, entry] of this.pending.entries()) {
          clearTimeout(entry.timer);
          entry.reject(new Error(`Python worker exited (code ${code}) while handling request ${id}`));
        }
        this.pending.clear();
      });
    });

    return this.readyPromise;
  }

  async call<T = unknown>(payload: Record<string, unknown>, timeoutMs = 45000): Promise<T> {
    await this.ensureStarted();
    if (!this.proc) {
      throw new Error('Python worker failed to start');
    }
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('Python worker request timed out'));
      }, timeoutMs);

      this.pending.set(id, {
        resolve: resolve as (v: unknown) => void,
        reject,
        timer,
      });
      this.proc!.stdin.write(`${JSON.stringify({ ...payload, id })}\n`);
    });
  }
}

async function createServer() {
  const app = express();
  const worker = new PythonWorkerPool();

  app.use(express.json({ limit: '64mb' }));

  app.get('/api/health', async (_req, res) => {
    try {
      const ping = await worker.call({ action: 'ping' }, 10000);
      res.json({ status: 'ok', worker: ping });
    } catch (err) {
      res.status(500).json({ status: 'error', error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.post('/api/analyze', async (req, res) => {
    try {
      const body = req.body ?? {};
      const preset = typeof body.preset === 'string' ? body.preset : undefined;
      const imageBase64 = typeof body.imageBase64 === 'string' ? body.imageBase64 : undefined;

      if (!preset && !imageBase64) {
        res.status(400).json({ error: 'Provide either preset or imageBase64.' });
        return;
      }

      const payload: Record<string, unknown> = {
        action: 'analyze',
        mode: body.mode ?? 'auto',
        sensitivity: Number(body.sensitivity ?? 0.65),
        minRegionRatio: Number(body.minRegionRatio ?? 0.08),
        bleedTrim: Number(body.bleedTrim ?? 2),
        contourGapPx: Number(body.contourGapPx ?? 8),
        rows: Number(body.rows ?? 3),
        cols: Number(body.cols ?? 3),
        gutterPx: Number(body.gutterPx ?? 0),
        uniformSize: Boolean(body.uniformSize),
        uniformMode: body.uniformMode ?? 'pad',
        targetWidth: body.targetWidth ?? null,
        targetHeight: body.targetHeight ?? null,
        format: body.format ?? 'png',
        includeTiles: body.includeTiles !== false,
        includeSourcePreview: Boolean(body.includeSourcePreview),
      };

      if (imageBase64) {
        payload.imageBase64 = imageBase64;
      } else if (preset === 'slides') {
        if (!fs.existsSync(SAMPLE_SLIDES_PATH)) {
          res.status(404).json({ error: 'Sample slides asset not found.' });
          return;
        }
        payload.imagePath = SAMPLE_SLIDES_PATH;
      } else {
        payload.preset = preset;
      }

      const result = await worker.call(payload, 45000);
      res.json(result);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  app.post('/api/benchmark', async (_req, res) => {
    try {
      const result = await worker.call({ action: 'benchmark' }, 60000);
      res.json(result);
    } catch (err) {
      res.status(500).json({ error: err instanceof Error ? err.message : String(err) });
    }
  });

  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.use('/examples', express.static(path.resolve(__dirname, 'public', 'examples')));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`PyCut Studio server listening on http://0.0.0.0:${PORT}`);
  });
}

createServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
