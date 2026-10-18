#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const { binaryPath } = require('..');

// Windows has no /proc/cpuinfo to check up front; a CPU below x86-64-v3
// exits with this NTSTATUS instead
const STATUS_ILLEGAL_INSTRUCTION = 0xC000001D;

let bin;
try {
  bin = binaryPath();
} catch (e) {
  console.error(`qwen-asr: ${e.message}.`);
  console.error('Build from source: https://github.com/nexustar/qwen-asr');
  process.exit(1);
}

const r = spawnSync(bin, process.argv.slice(2), { stdio: 'inherit', argv0: 'qwen-asr' });
if (r.error) {
  console.error(`qwen-asr: ${r.error.message}`);
  process.exit(1);
}
if (r.signal) process.kill(process.pid, r.signal);
if (r.status === STATUS_ILLEGAL_INSTRUCTION) {
  console.error('qwen-asr: this CPU lacks AVX2/FMA (x86-64-v3 required).');
  process.exit(1);
}
process.exit(r.status);
