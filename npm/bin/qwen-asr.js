#!/usr/bin/env node
'use strict';

const { spawnSync } = require('child_process');
const { binaryPath } = require('..');

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
process.exit(r.status);
