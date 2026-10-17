#!/usr/bin/env node
'use strict';

const { execFileSync, spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

let key = `${process.platform}-${process.arch}`;
// x64 Node under Rosetta: use the native arm64 binary
if (key === 'darwin-x64') {
  try {
    const out = execFileSync('/usr/sbin/sysctl', ['-in', 'sysctl.proc_translated'], { encoding: 'utf8' });
    if (out.trim() === '1') key = 'darwin-arm64';
  } catch (e) {}
}
const bin = path.join(__dirname, '..', 'prebuilds', key, 'qwen_asr');

if (!fs.existsSync(bin)) {
  console.error(`qwen-asr: no prebuilt binary for ${key}.`);
  console.error('Build from source: https://github.com/nexustar/qwen-asr');
  process.exit(1);
}

// Fail clearly instead of SIGILL below the build baseline
// (arm64: LSE atomics as a proxy for ARMv8.2)
const required = {
  'linux-x64': { field: 'flags', features: ['avx2', 'fma', 'f16c', 'bmi2'], level: 'x86-64-v3' },
  'linux-arm64': { field: 'Features', features: ['atomics'], level: 'ARMv8.2' },
}[key];
if (required) {
  let have = null;
  try {
    const re = new RegExp(`^${required.field}\\s*:(.*)$`, 'm');
    const m = re.exec(fs.readFileSync('/proc/cpuinfo', 'utf8'));
    if (m) have = new Set(m[1].trim().split(/\s+/));
  } catch (e) {}
  const missing = have ? required.features.filter(f => !have.has(f)) : [];
  if (missing.length) {
    console.error(`qwen-asr: this CPU lacks ${missing.join(', ')} (${required.level} required).`);
    console.error('Build from source: https://github.com/nexustar/qwen-asr');
    process.exit(1);
  }
}

const r = spawnSync(bin, process.argv.slice(2), { stdio: 'inherit', argv0: 'qwen-asr' });
if (r.error) {
  console.error(`qwen-asr: ${r.error.message}`);
  process.exit(1);
}
if (r.signal) process.kill(process.pid, r.signal);
process.exit(r.status);
