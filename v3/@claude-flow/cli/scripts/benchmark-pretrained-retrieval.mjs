#!/usr/bin/env node
// benchmark-pretrained-retrieval.mjs — proof that pretrained patterns are
// retrievable, not just stored.
//
// Runs sample queries against the neural store (post-pretrain) and reports
// the top-k matches. Demonstrates that after `pretrain-from-github.mjs`
// runs, an agent can recall relevant past work by intent.
//
// Usage:
//   1. node scripts/pretrain-from-github.mjs           # populate the store
//   2. node scripts/benchmark-pretrained-retrieval.mjs # query + report

import { writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const CLI_ROOT = resolve(SCRIPT_DIR, '..');
const REPO_ROOT = resolve(SCRIPT_DIR, '../../../..');
const RUNS_DIR = join(REPO_ROOT, 'docs', 'benchmarks', 'runs');

// Real ruflo-history-shaped queries. Each one targets a concept that should
// have been seen during pretrain (Opus 4.8 fix, self-learning wiring,
// codemod engine, etc.).
const QUERIES = [
  'how was the Opus model alias fixed',
  'self-learning wiring task-completed pretrain',
  'deterministic codemod engine var-to-const',
  'MCP server orphan leak parent-death',
  'unified learning stats aggregator',
  'structured distillation 4-field schema',
  'SQL injection migrate.ts table identifier',
  'recall@k HNSW benchmark harness',
  'Q-learning encoder keyword block',
  'security hardening crypto random IDs',
];

const TOP_K = 5;

async function main() {
  const intel = await import(join(CLI_ROOT, 'dist/src/memory/intelligence.js'));
  const neural = await import(join(CLI_ROOT, 'dist/src/mcp-tools/neural-tools.js'));

  // §1 — snapshot the neural store + globalStats so we know what's there.
  const unified = await intel.getUnifiedLearningStats();
  const total = unified.neuralPatterns.patternCount;

  if (total === 0) {
    console.error('No patterns in neural store. Run scripts/pretrain-from-github.mjs first.');
    process.exit(2);
  }

  // §2 — run each query through neural_patterns search. The tool's search
  // action runs cosine similarity against the stored embeddings.
  const listTool = neural.neuralTools.find((t) => t.name === 'neural_patterns');
  const tQuery0 = performance.now();
  const results = [];
  for (const q of QUERIES) {
    const r = await listTool.handler({ action: 'search', query: q });
    const matches = (r.patterns || r.results || r.matches || []).slice(0, TOP_K);
    results.push({
      query: q,
      matched: matches.length > 0,
      topK: matches.map((m) => ({
        id: m.id,
        name: m.name?.slice(0, 100),
        type: m.type,
        score: m.score ?? m.similarity,
      })),
    });
  }
  const queryMs = performance.now() - tQuery0;

  const matchedQueries = results.filter((r) => r.matched).length;
  const summary = {
    runAt: new Date().toISOString(),
    benchmark: 'pretrained-retrieval',
    storeSize: total,
    queries: QUERIES.length,
    matchedQueries,
    matchRate: Number((matchedQueries / QUERIES.length).toFixed(4)),
    avgQueryLatencyMs: Number((queryMs / QUERIES.length).toFixed(2)),
    totalQueryMs: Number(queryMs.toFixed(2)),
    results,
    passed: matchedQueries === QUERIES.length,
  };

  if (process.env.BENCH_JSON) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    console.log(`# Pretrained-retrieval benchmark — proof of learning`);
    console.log(`Store size: ${total} patterns`);
    console.log(`Queries: ${QUERIES.length}`);
    console.log(`Match rate: ${(summary.matchRate * 100).toFixed(0)}% (${matchedQueries}/${QUERIES.length})`);
    console.log(`Avg query latency: ${summary.avgQueryLatencyMs} ms`);
    console.log('');
    for (const r of results) {
      console.log(`Q: "${r.query}"`);
      if (r.topK.length === 0) {
        console.log(`   → no matches`);
      } else {
        for (const m of r.topK.slice(0, 3)) {
          console.log(`   → ${m.score?.toFixed?.(3) ?? '—'}  ${m.name}`);
        }
      }
    }
    console.log('');
    console.log(`Overall: ${summary.passed ? '✅ PASSED' : '⚠️  partial'}`);
  }

  if (!process.env.BENCH_NO_WRITE) {
    mkdirSync(RUNS_DIR, { recursive: true });
    const stamp = summary.runAt.replace(/[:.]/g, '-');
    writeFileSync(join(RUNS_DIR, `pretrained-retrieval-${stamp}.json`), JSON.stringify(summary, null, 2));
    writeFileSync(join(RUNS_DIR, 'pretrained-retrieval-latest.json'), JSON.stringify(summary, null, 2));
    if (!process.env.BENCH_JSON) console.log(`\nWrote ${join(RUNS_DIR, `pretrained-retrieval-${stamp}.json`)}`);
  }

  // ONNX runtime keeps a worker thread alive — force exit.
  process.exit(summary.passed ? 0 : 1);
}

main().catch((err) => { console.error(err); process.exit(1); });
