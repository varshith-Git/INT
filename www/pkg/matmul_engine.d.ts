/* tslint:disable */
/* eslint-disable */

/**
 * Load the quantized model from a Uint8Array (.bin file bytes).
 * Call this once after fetching the model file via fetch().
 * Returns true on success.
 */
export function init_model(bytes: Uint8Array): boolean;

/**
 * Run one forward pass. Returns logits for all 61 vocab tokens.
 *
 * token_ids: a sequence of up to 64 character token IDs.
 *            Feed the last N characters the user has typed (encoded as u8).
 *            The engine is stateless per call — it re-builds the KV cache
 *            from scratch each time (fine for short contexts, ≤64 chars).
 *
 * Returns Float32Array of length 61.
 */
export function predict(token_ids: Uint8Array): Float32Array;

/**
 * Model vocab size — always 61 for this build.
 */
export function vocab_size(): number;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly init_model: (a: number, b: number) => number;
    readonly predict: (a: number, b: number) => [number, number];
    readonly vocab_size: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
