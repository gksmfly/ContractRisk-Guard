// frontend/app/opengraph-image.tsx
import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Verilex — AI 계약 리스크 분석";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpengraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          background: "#0f1e3d",
          padding: "80px",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            marginBottom: 40,
          }}
        >
          <div
            style={{
              display: "flex",
              width: 56,
              height: 56,
              borderRadius: 14,
              background: "rgba(255,255,255,0.1)",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 28,
            }}
          >
            🛡
          </div>
          <div style={{ display: "flex", fontSize: 32, fontWeight: 700, color: "#ffffff" }}>
            Verilex
          </div>
        </div>
        <div
          style={{
            display: "flex",
            fontSize: 56,
            fontWeight: 800,
            color: "#ffffff",
            textAlign: "center",
            lineHeight: 1.3,
            maxWidth: 900,
          }}
        >
          약관 속 불리한 조항, AI가 법령 근거로 잡아냅니다
        </div>
        <div
          style={{
            display: "flex",
            fontSize: 24,
            color: "#e7edf3",
            marginTop: 32,
          }}
        >
          약관규제법 조문 · 판례 직접 인용 기반 리스크 분석
        </div>
      </div>
    ),
    { ...size }
  );
}
