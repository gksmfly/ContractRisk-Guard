// frontend/app/page.tsx
import { Navbar } from "@/components/Navbar";
import { Footer } from "@/components/Footer";
import { HeroSection } from "@/components/sections/HeroSection";
import { FeaturesSection } from "@/components/sections/FeaturesSection";
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
        <FeaturesSection />
        <HowToUseSection />
        <ExamplesSection />
        <TrustSection />
        <FAQSection />
        <CTASection />
      </main>
      <Footer />
    </div>
  );
}
