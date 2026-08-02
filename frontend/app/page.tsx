// frontend/app/page.tsx
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { ScrollReveal } from "@/components/ScrollReveal";
import { HeroSection } from "@/components/sections/HeroSection";
import { ProductPreviewSection } from "@/components/sections/ProductPreviewSection";
import { FeaturesSection } from "@/components/sections/FeaturesSection";
import { AudienceSection } from "@/components/sections/AudienceSection";
import { HowToUseSection } from "@/components/sections/HowToUseSection";
import { TrustSection } from "@/components/sections/TrustSection";
import { FAQSection } from "@/components/sections/FAQSection";
import { CTASection } from "@/components/sections/CTASection";
import { ExamplesSection } from "@/components/sections/ExamplesSection";

export default function Home() {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main>
        <HeroSection />
        <ScrollReveal>
          <ProductPreviewSection />
        </ScrollReveal>
        <ScrollReveal>
          <FeaturesSection />
        </ScrollReveal>
        <ScrollReveal>
          <AudienceSection />
        </ScrollReveal>
        <ScrollReveal>
          <HowToUseSection />
        </ScrollReveal>
        <ScrollReveal>
          <ExamplesSection />
        </ScrollReveal>
        {/* TrustSection 내부는 통계 카드마다 개별 ScrollReveal + CountUp을 씀 */}
        <TrustSection />
        <ScrollReveal>
          <FAQSection />
        </ScrollReveal>
        <ScrollReveal>
          <CTASection />
        </ScrollReveal>
      </main>
      <Footer />
    </div>
  );
}
