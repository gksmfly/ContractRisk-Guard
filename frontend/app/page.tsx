// frontend/app/page.tsx
import dynamic from "next/dynamic";
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { HeroSection } from "@/components/sections/HeroSection";
import { FeaturesSection } from "@/components/sections/FeaturesSection";
import { HowToUseSection } from "@/components/sections/HowToUseSection";
import { TrustSection } from "@/components/sections/TrustSection";
import { CTASection } from "@/components/sections/CTASection";

const ExamplesSection = dynamic(
  () => import("@/components/sections/ExamplesSection").then((m) => m.ExamplesSection),
  { ssr: false }
);

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        <HeroSection />
        <FeaturesSection />
        <HowToUseSection />
        <ExamplesSection />
        <TrustSection />
        <CTASection />
      </main>
      <Footer />
    </div>
  );
}
