// frontend/components/ScrollToTop.tsx
"use client";

import { useEffect, useState } from "react";
import { ArrowUp } from "lucide-react";

export function ScrollToTop() {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const handler = () => setVisible(window.scrollY > 400);
    window.addEventListener("scroll", handler, { passive: true });
    return () => window.removeEventListener("scroll", handler);
  }, []);

  const handleClick = () => window.scrollTo({ top: 0, behavior: "smooth" });

  return (
    <button
      onClick={handleClick}
      aria-label="페이지 맨 위로 이동"
      className={`fixed bottom-6 right-6 z-50 p-3 rounded-full
        bg-slate-800 border border-slate-700 text-slate-300
        hover:bg-blue-600 hover:border-blue-500 hover:text-white
        shadow-lg transition-all duration-300
        ${visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4 pointer-events-none"}`}
    >
      <ArrowUp className="h-4 w-4" />
    </button>
  );
}
