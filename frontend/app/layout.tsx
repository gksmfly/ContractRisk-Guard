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
  "계약 해지·책임제한 조항의 법적 리스크를 AI가 자동 탐지합니다. 약관규제법·민법 기반 판례 분석으로 고품질 리스크 레포트를 제공합니다.";

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
