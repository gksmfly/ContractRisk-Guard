// frontend/app/sitemap.ts
import type { MetadataRoute } from "next";

const siteUrl = process.env.NEXTAUTH_URL ?? "http://localhost:3000";

// 로그인 필요 페이지(대시보드)나 내부 비교용 데모 페이지는 공개 사이트맵에서 제외한다.
export default function sitemap(): MetadataRoute.Sitemap {
  const routes = ["", "/analyze", "/login"];
  return routes.map((route) => ({
    url: `${siteUrl}${route}`,
    lastModified: new Date(),
    changeFrequency: "weekly" as const,
    priority: route === "" ? 1 : 0.8,
  }));
}
