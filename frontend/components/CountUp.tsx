// frontend/components/CountUp.tsx
"use client";

import { useEffect, useRef, useState } from "react";

interface ParsedValue {
  num: number;
  decimals: number;
  suffix: string;
  hasComma: boolean;
}

function parseValue(raw: string): ParsedValue | null {
  const match = raw.match(/^([\d,]+(?:\.\d+)?)/);
  if (!match) return null;
  const numStr = match[1];
  const num = parseFloat(numStr.replace(/,/g, ""));
  const decimals = numStr.includes(".") ? numStr.split(".")[1].length : 0;
  return {
    num,
    decimals,
    suffix: raw.slice(numStr.length),
    hasComma: numStr.includes(","),
  };
}

function format(n: number, parsed: ParsedValue) {
  const body = parsed.hasComma
    ? Math.round(n).toLocaleString("ko-KR")
    : n.toFixed(parsed.decimals);
  return `${body}${parsed.suffix}`;
}

/**
 * 뷰포트에 들어오면 0에서 실제 값까지 세어 올라가는 통계 숫자.
 * "2,488건" · "4.98:1↑" · "6단계" 처럼 접미사가 붙은 값도 그대로 파싱해 유지한다.
 * 숫자로 시작하지 않는 값(예: "법령 원문")은 그대로 정적으로 표시된다.
 */
export function CountUp({
  value,
  duration = 1200,
  className,
}: {
  value: string;
  duration?: number;
  className?: string;
}) {
  const parsed = parseValue(value);
  const ref = useRef<HTMLSpanElement>(null);
  const [display, setDisplay] = useState(() => (parsed ? format(0, parsed) : value));

  useEffect(() => {
    const el = ref.current;
    if (!parsed || !el) {
      setDisplay(value);
      return;
    }
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      setDisplay(value);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return;
        observer.disconnect();
        const start = performance.now();
        const tick = (now: number) => {
          const progress = Math.min((now - start) / duration, 1);
          const eased = 1 - Math.pow(1 - progress, 3);
          if (progress < 1) {
            setDisplay(format(parsed.num * eased, parsed));
            requestAnimationFrame(tick);
          } else {
            setDisplay(value);
          }
        };
        requestAnimationFrame(tick);
      },
      { threshold: 0.4 }
    );
    observer.observe(el);
    return () => observer.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps -- parsed는 value에서 파생되므로 value만 의존성으로 충분
  }, [value, duration]);

  return (
    <span ref={ref} className={className}>
      {display}
    </span>
  );
}
