# frontend/

Next.js 14(App Router) 기반 랜딩 페이지 + 계약서 조항 분석 UI. `backend/api`(FastAPI)를 백엔드로 호출한다.

## 스택

Next.js 14 · React 18 · TypeScript · Tailwind CSS · Zustand · TanStack Query · Framer Motion · pdfjs-dist

## 디렉토리 구조

```
frontend/
├── app/
│   ├── page.tsx              # 랜딩 페이지
│   ├── layout.tsx / providers.tsx
│   ├── analyze/
│   │   ├── page.tsx
│   │   └── ContractAnalyzer.tsx   # 조항 분석 UI 메인 컴포넌트
│   └── api/                  # Next.js Route Handlers (BFF 계층)
│       ├── analyze/route.ts       # 키워드 기반 임시 분석(백엔드 미연동 폴백)
│       ├── analyze-full/route.ts  # FastAPI(/api/analyze) 프록시
│       └── analyze-pdf/route.ts   # FastAPI(/api/analyze-pdf) 프록시
├── components/
│   ├── Navbar.tsx / Footer.tsx / ScrollToTop.tsx
│   ├── sections/             # 랜딩 페이지 섹션 (Hero, Features, Examples, HowToUse, CTA, Trust)
│   └── ui/                   # shadcn 기반 프리미티브 (badge, card)
├── lib/utils.ts               # cn() 등 공통 유틸
└── types/index.ts             # Domain, RiskLevel, AnalyzeResult 등 공유 타입
```

## 백엔드 연동

`app/api/analyze-full`, `app/api/analyze-pdf`는 Route Handler에서 `FASTAPI_URL`(기본값 `http://localhost:8000`) 뒤의 `backend/api`(FastAPI 서버, `backend/api/server.py`)로 요청을 프록시한다. `app/api/analyze/route.ts`는 백엔드 미연동 상태에서 쓰던 키워드 기반 임시 분류 로직으로, 실제 판단은 FastAPI 쪽 KoELECTRA(Judgment Agent)가 담당한다.

## 실행

```bash
cd frontend
npm install
npm run dev        # http://localhost:3000, FASTAPI_URL로 backend/api 지정 필요
```

## 참고

- `frontend/.env*.local`, `node_modules/`, `.next/`는 `.gitignore`에 이미 등록됨.
- 백엔드 API 스키마는 `backend/api/schemas.py` 참고 — `types/index.ts`와 필드명을 맞춰야 함(현재는 수동 동기화, 자동 타입 생성기 없음).
