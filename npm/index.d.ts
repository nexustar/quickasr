/**
 * Absolute path of the prebuilt quickasr binary for this machine.
 *
 * Throws an Error with `code` QUICKASR_UNSUPPORTED_PLATFORM (no binary for
 * this OS / CPU architecture) or QUICKASR_UNSUPPORTED_CPU (the CPU is below
 * the build baseline) when none can run here.
 */
export function binaryPath(): string;
