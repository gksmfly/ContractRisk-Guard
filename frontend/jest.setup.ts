// frontend/jest.setup.ts
import "@testing-library/jest-dom";

// jsdom에는 Web Streams API(TextDecoder/TextEncoder/ReadableStream)가 없어서
// 분석 스트리밍(SSE) 코드가 참조하면 ReferenceError가 난다. 실제 브라우저와
// Next.js 런타임에는 전부 있으므로 테스트 환경에만 Node 구현을 채워 넣는다.
import { TextDecoder, TextEncoder } from "util";
import { ReadableStream } from "stream/web";

if (typeof global.TextDecoder === "undefined") {
  // @ts-expect-error - Node TextDecoder는 테스트 목적으로 충분히 호환된다
  global.TextDecoder = TextDecoder;
}
if (typeof global.TextEncoder === "undefined") {
  global.TextEncoder = TextEncoder;
}
if (typeof global.ReadableStream === "undefined") {
  // @ts-expect-error
  global.ReadableStream = ReadableStream;
}
