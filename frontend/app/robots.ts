// frontend/app/robots.ts
import type { MetadataRoute } from "next";

const siteUrl = process.env.NEXTAUTH_URL ?? "http://localhost:3000";

export default function robots(): MetadataRoute.Robots {
  return {
    rules: {
      userAgent: "*",
      allow: "/",
      disallow: ["/dashboard", "/design", "/analyze/"],
    },
    sitemap: `${siteUrl}/sitemap.xml`,
  };
}
