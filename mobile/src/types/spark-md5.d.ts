declare module 'spark-md5' {
  class SparkMD5ArrayBuffer {
    append(arr: globalThis.ArrayBuffer): void;
    end(raw?: boolean): string;
  }
  interface SparkMD5Static {
    ArrayBuffer: new () => SparkMD5ArrayBuffer;
  }
  const SparkMD5: SparkMD5Static;
  export default SparkMD5;
}
