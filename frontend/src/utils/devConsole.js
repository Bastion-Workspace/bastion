/**
 * Development-only console helpers. No-ops in production builds.
 * Use for trace and debug output; keep console.error / console.warn for real failures.
 */

const devEnabled =
  typeof import.meta !== 'undefined' &&
  import.meta.env &&
  !import.meta.env.PROD;

function noop() {}

export const devLog = devEnabled
  ? (...args) => {
      if (typeof console !== 'undefined' && console.log) {
        console.log(...args);
      }
    }
  : noop;

export const devDebug = devEnabled
  ? (...args) => {
      if (typeof console !== 'undefined' && console.debug) {
        console.debug(...args);
      }
    }
  : noop;

export const devInfo = devEnabled
  ? (...args) => {
      if (typeof console !== 'undefined' && console.info) {
        console.info(...args);
      }
    }
  : noop;
