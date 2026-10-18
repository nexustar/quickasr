/**
 * Absolute path of the prebuilt qwen_asr binary for this machine.
 *
 * Throws an Error with `code` QWEN_ASR_UNSUPPORTED_PLATFORM (no binary for
 * this OS / CPU architecture) or QWEN_ASR_UNSUPPORTED_CPU (the CPU is below
 * the build baseline) when none can run here.
 */
export function binaryPath(): string;
