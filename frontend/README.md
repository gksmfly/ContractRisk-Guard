# frontend/

Next.js 14(App Router) 기반 랜딩 페이지 + 계약서 조항 분석 UI. `backend/api`(FastAPI)를 백엔드로 호출한다.

## 스택

Next.js 14 · React 18 · TypeScript · Tailwind CSS · TanStack Query · Jest + React Testing Library

## 디렉토리 구조

```
frontend/
├── app/
│   ├── page.tsx              # 랜딩 페이지
│   ├── layout.tsx / providers.tsx
│   ├── error.tsx / not-found.tsx / loading.tsx   # App Router 에러/로딩 바운더리
│   ├── analyze/
│   │   ├── page.tsx
│   │   ├── ContractAnalyzer.tsx   # 조항 분석 UI 메인 컴포넌트 (사이드바+상세 패널, PDF 리포트 인쇄)
│   │   └── __tests__/             # Jest + RTL 테스트
│   └── api/                  # Next.js Route Handlers (BFF 계층)
│       ├── analyze-full/route.ts  # FastAPI(/api/analyze) 프록시
│       └── analyze-pdf/route.ts   # FastAPI(/api/analyze-pdf) 프록시
├── components/
│   ├── Navbar.tsx / Footer.tsx / ScrollToTop.tsx
│   ├── sections/             # 랜딩 페이지 섹션 (Hero, Features, HowToUse, Examples, Trust, FAQ, CTA)
│   └── ui/                   # shadcn 기반 프리미티브 (badge, card)
├── lib/
│   ├── utils.ts               # cn() 등 공통 유틸
│   └── config.ts              # FASTAPI_URL, 업로드 제한 등 공통 설정
├── jest.config.js / jest.setup.ts
└── types/index.ts             # Domain, RiskLevel, EvidenceSpan 등 공유 타입
```

`/analyze` 결과 화면은 `tailwind.config.ts`에 정의된 전용 팔레트(`paper`/`navy`/`seal`/`ochre`/`forest`)를 사용한다 — 법령 원문 인용을 강조하기 위한 문서형 스타일로, 랜딩 페이지의 기본 팔레트와는 의도적으로 분리했다.

## 백엔드 연동

`app/api/analyze-full`, `app/api/analyze-pdf`는 Route Handler에서 `lib/config.ts`의 `FASTAPI_URL`(기본값 `http://localhost:8000`) 뒤의 `backend/api`(FastAPI 서버, `backend/api/server.py`)로 요청을 프록시한다. 실제 판단은 FastAPI 쪽 KoELECTRA(Judgment Agent)가 담당한다.

## 실행

```bash
cd frontend
cp .env.example .env.local   # FASTAPI_URL 등 환경변수 설정
npm install
npm run dev        # http://localhost:3000
npm test            # Jest + RTL
npm run typecheck   # tsc --noEmit
```

## 참고

- `frontend/.env*.local`, `node_modules/`, `.next/`는 `.gitignore`에 이미 등록됨.
- 백엔드 API 스키마는 `backend/api/schemas.py` 참고 — `types/index.ts`와 필드명을 맞춰야 함(현재는 수동 동기화, 자동 타입 생성기 없음).
