'use strict';

// Locates the prebuilt qwen_asr binary for this machine. Used by the
// qwen-asr launcher, and by programs that spawn the binary themselves.

const { execFileSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Below the build baseline the binary dies with SIGILL, so check first
// (arm64: LSE atomics as a proxy for ARMv8.2)
const REQUIRED = {
  'linux-x64': { field: 'flags', features: ['avx2', 'fma', 'f16c', 'bmi2'], level: 'x86-64-v3' },
  'linux-arm64': { field: 'Features', features: ['atomics'], level: 'ARMv8.2' },
};

function platformKey() {
  const key = `${process.platform}-${process.arch}`;
  // x64 Node under Rosetta: use the native arm64 binary
  if (key === 'darwin-x64') {
    try {
      const out = execFileSync('/usr/sbin/sysctl', ['-in', 'sysctl.proc_translated'], { encoding: 'utf8' });
      if (out.trim() === '1') return 'darwin-arm64';
    } catch (e) {}
  }
  return key;
}

function unsupported(code, message) {
  const err = new Error(message);
  err.code = code;
  return err;
}

// Absolute path of the binary. Throws an Error with code
// QWEN_ASR_UNSUPPORTED_PLATFORM or QWEN_ASR_UNSUPPORTED_CPU when none can
// run here.
function binaryPath() {
  const key = platformKey();
  const bin = path.join(__dirname, 'prebuilds', key, 'qwen_asr');
  if (!fs.existsSync(bin)) {
    throw unsupported('QWEN_ASR_UNSUPPORTED_PLATFORM', `no prebuilt binary for ${key}`);
  }
  const required = REQUIRED[key];
  if (required) {
    let have = null;
    try {
      const re = new RegExp(`^${required.field}\\s*:(.*)$`, 'm');
      const m = re.exec(fs.readFileSync('/proc/cpuinfo', 'utf8'));
      if (m) have = new Set(m[1].trim().split(/\s+/));
    } catch (e) {}
    const missing = have ? required.features.filter(f => !have.has(f)) : [];
    if (missing.length) {
      throw unsupported('QWEN_ASR_UNSUPPORTED_CPU', `this CPU lacks ${missing.join(', ')} (${required.level} required)`);
    }
  }
  return bin;
}

exports.binaryPath = binaryPath;
