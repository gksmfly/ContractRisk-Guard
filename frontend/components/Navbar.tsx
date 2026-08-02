// frontend/components/Navbar.tsx
"use client";

import {
  Menu,
  X,
  ChevronDown,
  Upload,
  Cpu,
  FileText,
  AlertTriangle,
  CheckCircle2,
  Shield,
  Scale,
  BookOpen,
  HelpCircle,
  type LucideIcon,
} from "lucide-react";
import { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Logo } from "@/components/Logo";

interface PanelItem {
  icon: LucideIcon;
  label: string;
  desc: string;
  href: string;
}

interface NavLink {
  href: string;
  label: string;
  panel?: {
    items: PanelItem[];
    cta: { label: string; href: string };
  };
}

// 상단바 메가메뉴 콘텐츠 — 각 섹션에 실제로 있는 내용(단계·사례·통계·FAQ)을 그대로 요약해 재사용한다.
const NAV_LINKS: NavLink[] = [
  {
    href: "/#how-to-use",
    label: "이용 방법",
    panel: {
      items: [
        { icon: Upload, label: "STEP 01 · 계약서 업로드", desc: "텍스트 붙여넣기 또는 파일 업로드", href: "/#how-to-use" },
        { icon: Cpu, label: "STEP 02 · AI 조항 분석", desc: "공정위 시정조치 2,488건 학습 기반 판단", href: "/#how-to-use" },
        { icon: FileText, label: "STEP 03 · 리스크 레포트", desc: "위험도·법령 근거·하이라이트 확인", href: "/#how-to-use" },
      ],
      cta: { label: "지금 무료로 분석하기 →", href: "/analyze" },
    },
  },
  {
    href: "/#examples",
    label: "분석 사례",
    panel: {
      items: [
        { icon: AlertTriangle, label: "즉시 해지권 부여", desc: "고위험 · 약관규제법 §9 위반 소지", href: "/#examples" },
        { icon: AlertTriangle, label: "포괄적 면책 조항", desc: "고위험 · 약관규제법 §7 위반 소지", href: "/#examples" },
        { icon: CheckCircle2, label: "표준 해지 조항", desc: "저위험 · 정상 범주 예시", href: "/#examples" },
      ],
      cta: { label: "전체 분석 사례 보기 →", href: "/#examples" },
    },
  },
  {
    href: "/#trust",
    label: "신뢰 근거",
    panel: {
      items: [
        { icon: Shield, label: "학습 기반 2,488건", desc: "공정위 시정조치 사례 Ground Truth", href: "/#trust" },
        { icon: Scale, label: "참조 판례 1,995건", desc: "해지·책임제한 도메인 실제 법원 판례", href: "/#trust" },
        { icon: BookOpen, label: "적용 법령 4건", desc: "약관규제법 §7·§9, 민법 §543~766", href: "/#trust" },
      ],
      cta: { label: "신뢰 근거 자세히 보기 →", href: "/#trust" },
    },
  },
  {
    href: "/#faq",
    label: "자주 묻는 질문",
    panel: {
      items: [
        { icon: HelpCircle, label: "법적 효력이 있는 자문인가요?", desc: "아니요 — 참고용 리스크 진단 도구입니다", href: "/#faq" },
        { icon: HelpCircle, label: "입력한 계약서는 저장되나요?", desc: "아니요 — 분석 처리 후 저장하지 않습니다", href: "/#faq" },
        { icon: HelpCircle, label: "PDF 파일도 분석되나요?", desc: "네 — PDF·TXT 업로드와 붙여넣기 지원", href: "/#faq" },
      ],
      cta: { label: "전체 FAQ 보기 →", href: "/#faq" },
    },
  },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  const pathname = usePathname();

  useEffect(() => {
    const handler = () => {
      setScrolled(window.scrollY > 40);
      setOpenMenu(null);
    };
    handler();
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpenMenu(null);
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
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
        <nav
          className="hidden md:flex items-center gap-7"
          onMouseLeave={() => setOpenMenu(null)}
        >
          {NAV_LINKS.map((link) => (
            <div key={link.href} onMouseEnter={() => setOpenMenu(link.href)}>
              <a
                href={link.href}
                onFocus={() => setOpenMenu(link.href)}
                aria-haspopup={link.panel ? "true" : undefined}
                aria-expanded={link.panel ? openMenu === link.href : undefined}
                className={`group relative flex items-center gap-1 text-sm transition-colors py-1 rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 ${
                  scrolled
                    ? "text-slate-400 hover:text-white focus-visible:ring-white focus-visible:ring-offset-navy"
                    : "text-slate-600 hover:text-navy focus-visible:ring-navy focus-visible:ring-offset-white"
                }`}
              >
                {link.label}
                {link.panel && (
                  <ChevronDown
                    className={`h-3 w-3 transition-transform duration-200 ${
                      openMenu === link.href ? "rotate-180" : ""
                    }`}
                    aria-hidden
                  />
                )}
                <span
                  className={`absolute left-0 -bottom-0.5 h-px w-0 transition-all duration-300 group-hover:w-full ${
                    scrolled ? "bg-navy-soft" : "bg-navy"
                  }`}
                />
              </a>
            </div>
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

      {/* Mega menu panels (desktop only) */}
      {NAV_LINKS.map((link) =>
        link.panel ? (
          <div
            key={`panel-${link.href}`}
            onMouseEnter={() => setOpenMenu(link.href)}
            onMouseLeave={() => setOpenMenu(null)}
            className={`hidden md:block absolute left-0 right-0 top-full bg-white border-t border-slate-100 shadow-xl overflow-hidden transition-all duration-200 ease-out ${
              openMenu === link.href ? "opacity-100 max-h-96" : "opacity-0 max-h-0 pointer-events-none"
            }`}
          >
            <div className="max-w-5xl mx-auto px-4 py-6 grid grid-cols-3 gap-3">
              {link.panel.items.map((item) => (
                <Link
                  key={item.label}
                  href={item.href}
                  onClick={() => setOpenMenu(null)}
                  className="flex items-start gap-3 p-3 rounded-xl hover:bg-slate-50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2"
                >
                  <div className="p-2 rounded-lg bg-navy-soft shrink-0">
                    <item.icon className="h-4 w-4 text-navy" aria-hidden />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-slate-900">{item.label}</p>
                    <p className="text-xs text-slate-500 mt-0.5">{item.desc}</p>
                  </div>
                </Link>
              ))}
            </div>
            <div className="border-t border-slate-100 px-4 py-3">
              <div className="max-w-5xl mx-auto">
                <Link
                  href={link.panel.cta.href}
                  onClick={() => setOpenMenu(null)}
                  className="text-sm font-semibold text-navy hover:underline rounded focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy focus-visible:ring-offset-2"
                >
                  {link.panel.cta.label}
                </Link>
              </div>
            </div>
          </div>
        ) : null
      )}

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
