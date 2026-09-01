// frontend/app/layout.tsx
import type { Metadata } from "next";
import { Noto_Sans_KR } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { ScrollToTop } from "@/components/ScrollToTop";

// 사이트 콘텐츠가 대부분 한글이라 라틴 전용 폰트(Inter) 대신 한글 글리프를 포함한 폰트를 쓴다.
const notoSansKr = Noto_Sans_KR({ subsets: ["latin"], weight: ["400", "500", "700", "900"] });

const siteUrl = process.env.NEXTAUTH_URL ?? "http://localhost:3000";
const title = "Verilex — AI 계약 리스크 분석";
const description =
  // "자동 탐지"·"고품질 리스크 레포트"를 뺐다 — 둘 다 완전성을 함의하는데 조항 단위
  // 재현이 78%다(5건 중 1건은 놓친다). 측정된 성능과 광고 문구가 어긋나면 성능 문제가
  // 아니라 신뢰 문제가 된다. 약관규제법을 다루는 도구라면 특히.
  "약관규제법 제6~14조 기준으로 확인이 필요한 계약 조항을 먼저 짚어드립니다. 법률 자문이 아닙니다.";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title,
  description,
  keywords: ["계약 리스크", "약관규제법", "계약 해지", "책임제한", "AI 법률 분석"],
  openGraph: {
    title,
    description,
    url: siteUrl,
    siteName: "Verilex",
    locale: "ko_KR",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title,
    description,
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body className={`${notoSansKr.className} antialiased`}>
        <Providers>
          {children}
          <ScrollToTop />
        </Providers>
      </body>
    </html>
  );
}
