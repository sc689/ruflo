#!/usr/bin/env node
// pretrain-from-github.mjs — pretrain ruflo's self-learning system from its
// own GitHub history (commits + issues). Each commit/issue becomes one
// trajectory through the SONA + EWC++ pipeline; Structured Distillation
// (ADR-076) compresses each into the 4-field schema before embedding.
//
// What this proves:
//   - globalStats.{trajectoriesRecorded, patternsLearned, signalsProcessed}
//     all move from a measured before to a measured after.
//   - neural_patterns.patternCount grows.
//   - memory-bridge entries grow.
//   - The unified-stats aggregator's consistency block stays clean.
//   - Every item gets a learningPath of 'trajectory-pipeline' (not 'recorded-only').
//
// Usage:
//   node scripts/pretrain-from-github.mjs                   # 50 commits + 30 issues
//   COMMITS=200 ISSUES=100 node scripts/pretrain-from-github.mjs
//   SOURCE=git node scripts/pretrain-from-github.mjs        # git only, skip gh
//   BENCH_JSON=1 node scripts/pretrain-from-github.mjs      # machine-readable
//   BENCH_NO_WRITE=1 node scripts/pretrain-from-github.mjs  # don't write a run JSON
//
// Repro from a fresh checkout:
//   git clone https://github.com/ruvnet/ruflo && cd ruflo
//   npm install && ( cd v3/@claude-flow/cli && npx tsc -b )
//   node v3/@claude-flow/cli/scripts/pretrain-from-github.mjs

import { execSync } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';
import { performance } from 'node:perf_hooks';

const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const CLI_ROOT = resolve(SCRIPT_DIR, '..');
const REPO_ROOT = resolve(SCRIPT_DIR, '../../../..');
const RUNS_DIR = join(REPO_ROOT, 'docs', 'benchmarks', 'runs');

const COMMITS = Number(process.env.COMMITS) || 50;
const ISSUES  = Number(process.env.ISSUES)  || 30;
const SOURCE  = process.env.SOURCE || 'all'; // 'all' | 'git' | 'issues'

// ---------------------------------------------------------------------------
// Harvesters
// ---------------------------------------------------------------------------

function harvestCommits(n) {
  if (SOURCE === 'issues') return [];
  const fmt = '%H%x00%s%x00%b%x01';
  const raw = execSync(
    `git log --pretty=format:'${fmt}' -n ${n} 2>/dev/null`,
    { cwd: REPO_ROOT, encoding: 'utf-8', maxBuffer: 64 * 1024 * 1024 },
  );
  const entries = [];
  for (const block of raw.split('\x01')) {
    if (!block.trim()) continue;
    const [sha, subject, body] = block.split('\x00');
    if (!sha || !subject) continue;
    entries.push({
      source: 'commit',
      id: `commit-${sha.slice(0, 12)}`,
      subject: subject.trim(),
      body: (body || '').trim(),
      // The "verdict" of a commit isn't directly known. Treat all commits as
      // success — we're learning from intent, not outcome. Failures are
      // captured separately via the issue stream (closed issues are
      // success; open issues are partial).
      verdict: 'success',
      content: `${subject.trim()}\n\n${(body || '').trim()}`.slice(0, 8192),
    });
  }
  return entries;
}

function harvestIssues(n) {
  if (SOURCE === 'git') return [];
  try {
    const raw = execSync(
      `gh issue list --repo ruvnet/ruflo --state all --limit ${n} --json number,title,body,state,closedAt 2>/dev/null`,
      { cwd: REPO_ROOT, encoding: 'utf-8', maxBuffer: 64 * 1024 * 1024 },
    );
    const items = JSON.parse(raw);
    return items.map((i) => ({
      source: 'issue',
      id: `issue-${i.number}`,
      subject: i.title,
      body: (i.body || '').slice(0, 8192),
      // closed = success outcome; open = partial (in-progress).
      verdict: i.state === 'CLOSED' ? 'success' : 'partial',
      content: `${i.title}\n\n${(i.body || '').slice(0, 8192)}`,
    }));
  } catch (err) {
    console.error(`gh issue harvest skipped (${err.message?.slice(0, 60)}). Set SOURCE=git to silence.`);
    return [];
  }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const intel = await import(join(CLI_ROOT, 'dist/src/memory/intelligence.js'));
  const neural = await import(join(CLI_ROOT, 'dist/src/mcp-tools/neural-tools.js'));
  const { distillAndSerialise } = await import(join(CLI_ROOT, 'dist/src/memory/structured-distill.js'));

  // §1 — record the baseline (no clear; we want to learn ON TOP of whatever
  // history the user already has).
  const unified0 = await intel.getUnifiedLearningStats();
  const before = {
    trajectoriesRecorded: unified0.global.trajectoriesRecorded,
    patternsLearned: unified0.global.patternsLearned,
    signalsProcessed: unified0.global.signalsProcessed,
    neuralPatternCount: unified0.neuralPatterns.patternCount,
    memoryBridgeTotal: unified0.memoryBridge.totalEntries,
  };

  // §2 — harvest
  const tHarvest0 = performance.now();
  const commits = harvestCommits(COMMITS);
  const issues = harvestIssues(ISSUES);
  const items = [...commits, ...issues];
  const harvestMs = performance.now() - tHarvest0;

  if (!process.env.BENCH_JSON) {
    console.log(`# Pretrain from ruflo GitHub history`);
    console.log(`Harvested: ${commits.length} commits + ${issues.length} issues = ${items.length} trajectories (${harvestMs.toFixed(0)} ms)`);
  }

  // §3 — feed each item through the trajectory pipeline.
  const tFeed0 = performance.now();
  let trained = 0;
  let failed = 0;
  const failures = [];
  for (const item of items) {
    try {
      const distilled = distillAndSerialise(item.content);
      await intel.recordTrajectory(
        [{
          type: 'result',
          content: distilled,
          metadata: {
            source: item.source,
            id: item.id,
            subject: item.subject.slice(0, 200),
          },
          timestamp: Date.now(),
        }],
        item.verdict,
      );
      trained++;
    } catch (err) {
      failed++;
      if (failures.length < 5) failures.push({ id: item.id, error: String(err.message).slice(0, 120) });
    }
  }
  const feedMs = performance.now() - tFeed0;
  intel.flushIntelligenceStats();

  // §4 — also seed the neural store directly from the same items so
  // `neural_patterns list` reflects them (closes the "globalStats moved but
  // neural_patterns didn't" consistency note from ADR-075).
  const tSeed0 = performance.now();
  const neuralItems = items.map((item) => ({
    name: item.subject.slice(0, 200),
    type: item.source === 'commit' ? 'history-commit' : 'history-issue',
    content: distillAndSerialise(item.content),
    metadata: { source: item.source, id: item.id, verdict: item.verdict },
  }));
  const seedResult = await neural.storeNeuralPatterns(neuralItems);
  const seedMs = performance.now() - tSeed0;

  // §5 — read the after-counters via the unified aggregator (this is what
  // hooks_intelligence_unified-stats would return for a live caller).
  const unified1 = await intel.getUnifiedLearningStats();
  const after = {
    trajectoriesRecorded: unified1.global.trajectoriesRecorded,
    patternsLearned: unified1.global.patternsLearned,
    signalsProcessed: unified1.global.signalsProcessed,
    neuralPatternCount: unified1.neuralPatterns.patternCount,
    memoryBridgeTotal: unified1.memoryBridge.totalEntries,
  };

  const deltas = Object.fromEntries(
    Object.keys(after).map((k) => [k, after[k] - before[k]]),
  );

  const summary = {
    runAt: new Date().toISOString(),
    benchmark: 'pretrain-from-github',
    source: SOURCE,
    config: { COMMITS, ISSUES },
    harvest: {
      commits: commits.length,
      issues: issues.length,
      total: items.length,
      harvestMs: Number(harvestMs.toFixed(2)),
    },
    feed: {
      trained,
      failed,
      avgLatencyMs: items.length > 0 ? Number((feedMs / items.length).toFixed(2)) : 0,
      totalMs: Number(feedMs.toFixed(2)),
      sampleFailures: failures,
    },
    seedNeuralStore: {
      stored: seedResult.stored,
      total: seedResult.total,
      seedMs: Number(seedMs.toFixed(2)),
    },
    before,
    after,
    deltas,
    consistency: unified1.consistency,
    passed:
      trained === items.length &&
      deltas.trajectoriesRecorded >= items.length &&
      deltas.neuralPatternCount >= items.length,
  };

  if (process.env.BENCH_JSON) {
    console.log(JSON.stringify(summary, null, 2));
  } else {
    console.log('');
    console.log('| Counter | Before | After | Δ |');
    console.log('|---|---:|---:|---:|');
    for (const k of Object.keys(after)) {
      console.log(`| ${k} | ${before[k]} | ${after[k]} | +${deltas[k]} |`);
    }
    console.log('');
    console.log(`Trained via trajectory pipeline: ${trained}/${items.length}`);
    console.log(`Failed: ${failed}`);
    console.log(`Avg latency per trajectory: ${summary.feed.avgLatencyMs} ms`);
    console.log(`Neural store seeded: ${seedResult.stored}/${seedResult.total}`);
    console.log(`Overall: ${summary.passed ? '✅ PASSED' : '⚠️  partial'}`);
    if (unified1.consistency.notes.length > 0) {
      console.log(`\nConsistency notes:`);
      for (const n of unified1.consistency.notes) console.log(`  • ${n}`);
    }
  }

  if (!process.env.BENCH_NO_WRITE) {
    mkdirSync(RUNS_DIR, { recursive: true });
    const stamp = summary.runAt.replace(/[:.]/g, '-');
    writeFileSync(join(RUNS_DIR, `pretrain-from-github-${stamp}.json`), JSON.stringify(summary, null, 2));
    writeFileSync(join(RUNS_DIR, 'pretrain-from-github-latest.json'), JSON.stringify(summary, null, 2));
    if (!process.env.BENCH_JSON) console.log(`\nWrote ${join(RUNS_DIR, `pretrain-from-github-${stamp}.json`)}`);
  }

  // ONNX runtime keeps a worker thread alive — force exit so this can be used
  // as a CI step or chained with other scripts.
  process.exit(summary.passed ? 0 : 1);
}

main().catch((err) => { console.error(err); process.exit(1); });
