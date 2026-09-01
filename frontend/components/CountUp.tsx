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

    // 화면에 들어올 때마다 다시 재생한다 — disconnect()로 한 번만 트는 대신, 나갈 때
    // 0으로 되돌려서 다음에 다시 스크롤해 들어와도 눈에 보이게 다시 올라가게 한다.
    // token으로 이전 애니메이션 프레임 루프를 무효화해 빠르게 들락날락해도 안 꼬이게 한다.
    let token = 0;

    const animate = () => {
      const myToken = ++token;
      const start = performance.now();
      const tick = (now: number) => {
        if (myToken !== token) return;
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
    };

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          animate();
        } else {
          token++; // 진행 중이던 애니메이션 프레임을 무효화
          setDisplay(format(0, parsed));
        }
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
