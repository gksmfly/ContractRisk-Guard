// frontend/components/Navbar.tsx
"use client";

import { Shield, Menu, X } from "lucide-react";
import { useState, useEffect } from "react";
import Link from "next/link";

const NAV_LINKS = [
  { href: "/#how-to-use", label: "이용 방법" },
  { href: "/#examples",   label: "분석 사례" },
  { href: "/#trust",      label: "신뢰 근거" },
  { href: "/#faq",        label: "자주 묻는 질문" },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 40);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  const scrollToTop = () => {
    setMenuOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 bg-[#0a0f1e]
        ${scrolled ? "shadow-md border-b border-white/5" : ""}`}
    >
      <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
        <button
          onClick={scrollToTop}
          aria-label="페이지 최상단으로 이동"
          className="flex items-center gap-2.5 group"
        >
          <div className="p-1.5 bg-blue-500/10 rounded-lg border border-blue-500/20 group-hover:bg-blue-500/20 transition-colors">
            <Shield className="h-4 w-4 text-blue-400" />
          </div>
          <span className="font-bold text-white text-sm group-hover:text-blue-300 transition-colors">
            Contract<span className="text-blue-400">Risk</span> Guard
          </span>
        </button>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-7">
          {NAV_LINKS.map((link) => (
            <a key={link.href} href={link.href}
              className="text-sm text-slate-400 hover:text-white transition-colors">
              {link.label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Link href="/analyze"
            className="text-xs px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors font-medium">
            계약서 분석 시작
          </Link>
          {/* Mobile hamburger */}
          <button
            onClick={() => setMenuOpen((v) => !v)}
            aria-label={menuOpen ? "메뉴 닫기" : "메뉴 열기"}
            aria-expanded={menuOpen}
            className="md:hidden p-2 text-slate-400 hover:text-white transition-colors"
          >
            {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile dropdown */}
      {menuOpen && (
        <div className="md:hidden border-t border-white/5 bg-[#0a0f1e] px-4 pb-4">
          <nav className="flex flex-col gap-1 pt-2">
            {NAV_LINKS.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setMenuOpen(false)}
                className="text-sm text-slate-400 hover:text-white py-2 px-2 rounded-lg hover:bg-white/5 transition-colors"
              >
                {link.label}
              </a>
            ))}
          </nav>
        </div>
      )}
    </header>
  );
}
