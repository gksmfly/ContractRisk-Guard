// frontend/components/Navbar.tsx
"use client";

import { Menu, X } from "lucide-react";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Logo } from "@/components/Logo";

const NAV_LINKS = [
  { href: "/#how-to-use", label: "이용 방법" },
  { href: "/#examples", label: "분석 사례" },
  { href: "/#trust", label: "신뢰 근거" },
  { href: "/#faq", label: "자주 묻는 질문" },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    const handler = () => setScrolled(window.scrollY > 40);
    handler();
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  const handleLogoClick = (e: React.MouseEvent) => {
    setMenuOpen(false);
    if (pathname === "/") {
      e.preventDefault();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }
  };

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 backdrop-blur-md ${
        scrolled
          ? "bg-navy/90 backdrop-blur-xl shadow-lg shadow-black/20 border-b border-white/10"
          : "bg-white/70 border-b border-transparent"
      }`}
    >
      <div className="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
        <Link
          href="/"
          onClick={handleLogoClick}
          aria-label="Verilex 홈으로 이동"
          className={`flex items-center gap-2.5 group rounded-lg focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 ${
            scrolled ? "focus-visible:ring-white focus-visible:ring-offset-navy" : "focus-visible:ring-navy focus-visible:ring-offset-white"
          }`}
        >
          <div
            className={`p-1.5 rounded-lg border transition-all group-hover:scale-105 ${
              scrolled
                ? "bg-white/10 border-navy-soft/20 group-hover:bg-white/20"
                : "bg-navy-soft border-navy/10 group-hover:bg-navy-soft/70"
            }`}
          >
            <Logo className={`h-4 w-4 transition-colors ${scrolled ? "text-navy-soft" : "text-navy"}`} />
          </div>
          <span
            className={`font-bold text-sm transition-colors ${
              scrolled ? "text-white group-hover:text-navy-soft" : "text-slate-900 group-hover:text-navy"
            }`}
          >
            Veri<span className={scrolled ? "text-navy-soft" : "text-navy"}>lex</span>
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-7">
          {NAV_LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className={`group relative text-sm transition-colors py-1 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 ${
                scrolled
                  ? "text-slate-400 hover:text-white focus-visible:ring-white focus-visible:ring-offset-navy"
                  : "text-slate-600 hover:text-navy focus-visible:ring-navy focus-visible:ring-offset-white"
              }`}
            >
              {link.label}
              <span
                className={`absolute left-0 -bottom-0.5 h-px w-0 transition-all duration-300 group-hover:w-full ${
                  scrolled ? "bg-navy-soft" : "bg-navy"
                }`}
              />
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <Link
            href="/login"
            className={`hidden sm:inline-block text-xs px-3 py-2 rounded transition-colors font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 ${
              scrolled
                ? "text-slate-300 hover:text-white focus-visible:ring-white focus-visible:ring-offset-navy"
                : "text-slate-600 hover:text-navy focus-visible:ring-navy focus-visible:ring-offset-white"
            }`}
          >
            로그인
          </Link>
          <Link
            href="/analyze"
            className={`text-xs px-4 py-2 bg-navy hover:opacity-90 text-white rounded-lg transition-opacity font-medium focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2 ${
              scrolled ? "ring-1 ring-white/15 focus-visible:ring-offset-navy" : "focus-visible:ring-offset-white"
            }`}
          >
            무료로 시작
          </Link>
          {/* Mobile hamburger */}
          <button
            onClick={() => setMenuOpen((v) => !v)}
            aria-label={menuOpen ? "메뉴 닫기" : "메뉴 열기"}
            aria-expanded={menuOpen}
            className={`md:hidden p-2 rounded-lg transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 ${
              scrolled
                ? "text-slate-400 hover:text-white focus-visible:ring-white focus-visible:ring-offset-navy"
                : "text-slate-600 hover:text-navy focus-visible:ring-navy focus-visible:ring-offset-white"
            }`}
          >
            {menuOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile dropdown */}
      {menuOpen && (
        <div
          className={`md:hidden border-t px-4 pb-4 transition-colors ${
            scrolled ? "border-white/5 bg-navy" : "border-slate-200 bg-white/95 backdrop-blur-md"
          }`}
        >
          <nav className="flex flex-col gap-1 pt-2">
            {NAV_LINKS.map((link) => (
              <a
                key={link.href}
                href={link.href}
                onClick={() => setMenuOpen(false)}
                className={`text-sm py-2 px-2 rounded-lg transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 ${
                  scrolled
                    ? "text-slate-400 hover:text-white hover:bg-white/5 focus-visible:ring-white focus-visible:ring-offset-navy"
                    : "text-slate-600 hover:text-navy hover:bg-navy-soft focus-visible:ring-navy focus-visible:ring-offset-white"
                }`}
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
