// frontend/app/layout.tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { ScrollToTop } from "@/components/ScrollToTop";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "ContractRisk Guard — AI 계약 리스크 분석",
  description:
    "계약 해지·책임제한 조항의 법적 리스크를 AI가 자동 탐지합니다. 약관규제법·민법 기반 판례 분석으로 고품질 리스크 레포트를 제공합니다.",
  keywords: ["계약 리스크", "약관규제법", "계약 해지", "책임제한", "AI 법률 분석"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body className={`${inter.className} antialiased`}>
        <Providers>
          {children}
          <ScrollToTop />
        </Providers>
      </body>
    </html>
  );
}
