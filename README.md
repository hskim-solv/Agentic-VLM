# BidMate Agent
**RFP 문서 이해를 위한 Agentic RAG 시스템**  
**1인 개발 · 문서 검색/비교/근거 검증 · 실무형 QA 시스템**

> 긴 제안요청서(RFP)에서 예산, 요구사항, 제출 조건, 사업 목적 같은 핵심 정보를 빠르게 찾아야 하는 문제를 해결하기 위해,  
> 질문을 해석하고, 관련 문서를 좁히고, 근거를 검증하며 답변하는 **agentic document intelligence system** 을 설계·구현한 프로젝트입니다.

---

## TL;DR

- **문제**: 공공/기업 RFP는 길고 복잡하며, 실무자는 여러 문서를 비교하며 핵심 조건을 빠르게 파악해야 합니다.
- **해결**: PDF/HWP 문서를 인덱싱하고, 질문 유형을 분석한 뒤, 메타데이터 기반 검색 + dense retrieval + reranking + evidence verification으로 답변하는 시스템을 구현했습니다.
- **차별점**: 단순 챗봇이 아니라 **query planning / grounded answer / retry loop** 를 포함한 **Agentic RAG** 구조를 적용했습니다.
- **검증**: 단일 문서 추출, 다문서 비교, 후속 질문, 부재 정보 판별을 포함한 평가셋으로 성능을 검증했습니다.
- **개발 범위**: 문제 정의부터 데이터 처리, retrieval, generation, 평가, 보고서 정리까지 **1인 개발**로 수행했습니다.

---

## 1. 프로젝트 소개

RFP(Request for Proposal, 제안요청서)는 문서 길이가 길고 형식이 제각각이며, 실무에서는 단순 검색보다 **정확한 근거 기반 요약과 비교**가 더 중요합니다.

예를 들어 아래와 같은 질문에 빠르게 답할 수 있어야 합니다.

- “이 사업의 **예산**이 얼마인지”
- “제출 방식과 일정이 어떻게 되는지”
- “기관 A의 사업과 기관 B의 사업이 **무엇이 다른지**”
- “이 문서에 **정말** AI 기반 요구사항이 있는지”
- “방금 답변한 사업에서 **세부 요구사항만 더 자세히** 알려달라”

즉, 필요한 것은 단순 keyword search가 아니라 다음을 수행하는 시스템입니다.

1. 질문 의도를 파악하고
2. 관련 문서를 좁히고
3. 근거를 수집하고
4. 비교/요약하고
5. 근거가 부족하면 모른다고 답하는 것

본 프로젝트는 이 과정을 자동화하는 **document decision assistant** 를 목표로 했습니다.

---

## 2. 문제 정의

RFP 문서 QA는 일반적인 단일 문서 검색보다 더 어렵습니다.

### 주요 난점
- 문서 포맷이 혼합됨 (`PDF`, `HWP`)
- 기관명 / 사업명이 사용자 질문과 정확히 일치하지 않을 수 있음
- 하나의 질문에 여러 문서를 비교해야 하는 경우가 많음
- 후속 질문에서 이전 대화 맥락을 유지해야 함
- 문서에 없는 내용은 추측하지 않고 **모른다고 답해야 함**

이 프로젝트의 핵심은 다음 네 가지를 함께 만족하는 것입니다.

- **문서 검색**
- **근거 수집**
- **비교/추론**
- **부재 판단**

---

## 3. 프로젝트 목표

본 프로젝트의 목표는 다음과 같습니다.

1. **RFP 문서 기반 질의응답 시스템 구현**
2. **PDF/HWP 문서 파싱 및 메타데이터 활용**
3. **단일 문서 추출 / 다문서 비교 / 후속 질문 처리**
4. **근거 기반 응답 및 hallucination 억제**
5. **정량·정성 평가를 통한 성능 검증**
6. **재현 가능한 코드/실험/로그 구조 설계**

---

## 4. 핵심 기능

### 4.1 문서 수집 및 인덱싱
- PDF/HWP 문서 로딩
- 문서별 메타데이터 정제
- 청킹(chunking) 및 임베딩 생성
- Vector DB 구축

### 4.2 질문 분석
- 질문 유형 분류
  - 단일 문서 추출형
  - 단일 문서 세부 탐색형
  - 다문서 비교형
  - 후속 질문형
  - 부재 확인형

### 4.3 Agentic Retrieval
- 메타데이터 기반 후보 문서 축소
- dense retrieval
- reranking
- 필요 시 fallback / retry

### 4.4 답변 생성
- 근거 문서/페이지 기반 답변 생성
- 비교형 질문은 표/항목 중심으로 정리
- 문서에 없는 내용은 추정하지 않고 `확인되지 않음` 처리

### 4.5 검증 및 로깅
- 답변-근거 일치 여부 점검
- retrieval 실패 시 재검색
- 질의별 실행 로그 저장
  - query type
  - selected docs
  - retrieved chunks
  - retry count
  - latency
  - final citation

---

## 5. 시스템 아키텍처

```text
User Query
  ↓
Query Analyzer
  - 질문 유형 분류
  - 기관명 / 사업명 / 주제 추출
  ↓
Planner
  - 검색 전략 결정
  - 메타데이터 필터링 조건 설정
  ↓
Retriever
  - metadata filtering
  - dense retrieval
  - reranking
  ↓
Evidence Aggregator
  - 근거 청크 병합
  - 페이지/섹션 정보 정리
  ↓
Answer Generator
  - 근거 기반 답변 생성
  - 비교/요약 형식 제어
  ↓
Verifier
  - 답변-근거 정합성 확인
  - 부족하면 retry
  ↓
Final Response
```

### 설계 포인트
이 프로젝트는 단순한 `retrieve -> generate` 구조가 아니라,

- **질문을 먼저 해석**하고
- **검색 계획을 세운 뒤**
- **근거를 확인하며**
- **필요 시 재시도하는**

형태로 설계하여 **agentic document intelligence system** 에 가깝게 구현했습니다.

---

## 6. 시스템 설계 의사결정

### 6.1 Metadata-first Retrieval
100개 수준의 문서를 전역 벡터 검색만으로 처리하면, 질문과 직접 관련 없는 문서가 retrieval 상위에 섞일 가능성이 높습니다.  
따라서 기관명, 사업명, 공고 식별 정보 등 **메타데이터를 먼저 활용해 후보군을 좁히고**, 이후 본문 검색을 수행하도록 설계했습니다.

### 6.2 Evidence-grounded Answering
실무형 QA 시스템에서는 “그럴듯한 답”보다 “근거 있는 답”이 더 중요합니다.  
그래서 최종 답변에는 가능한 한 근거 문서, 페이지/섹션, 관련 스니펫이 함께 연결되도록 설계했습니다.

### 6.3 Abstention over Hallucination
문서에 없는 내용을 모델이 일반 지식으로 보완하면 실무에서 오히려 치명적일 수 있습니다.  
따라서 근거가 충분하지 않은 경우에는 답을 지어내기보다 **확인되지 않음 / 근거 부족**으로 응답하도록 제한했습니다.

### 6.4 Retry when Retrieval Fails
초기 retrieval 결과가 부족하거나 비교 질문에서 한쪽 문서만 잡히는 경우가 있어,  
질문 재작성, top-k 조정, 필터 완화, rerank 재실행 등 **retry 루프**를 추가했습니다.

---

## 7. 데이터 구성

- 입력 데이터: 실제 RFP 문서 + 문서별 메타데이터
- 문서 포맷: `PDF`, `HWP`
- 주요 활용 메타데이터:
  - 발주 기관
  - 사업명
  - 공고 식별 정보
  - 기타 문서 구분 정보

### 데이터 공개 관련 주의
원본 RFP 문서는 외부 공유가 제한되어 있어 본 저장소에 포함하지 않았습니다.  
대신 아래를 공개합니다.

- 재현 가능한 코드
- 시스템 설계
- 평가 방식
- 로그 구조
- 2차 가공 결과 및 분석

---

## 8. 처리 가능한 질문 유형

### 8.1 단일 문서 추출형
- 특정 사업의 목적, 예산, 일정, 제출 조건 등 핵심 정보 추출

### 8.2 단일 문서 세부 탐색형
- 특정 요구사항, 기능, 운영 조건, 성능 조건 등 세부 항목 탐색

### 8.3 다문서 비교형
- 기관 A와 기관 B의 유사 사업 비교
- 기능, 성능, 요구사항, 제안 범위 차이 정리

### 8.4 후속 질문형
- 이전 답변의 문맥을 유지한 상태에서 특정 항목만 추가로 질의

### 8.5 부재 정보 판별형
- 문서 내에 명시적으로 존재하지 않는 정보를 정확히 “없음/확인 불가”로 판별

---

## 9. 평가 설계

본 프로젝트는 “데모가 돌아간다” 수준이 아니라,  
**어떤 질문에 얼마나 잘 답하는지**를 검증하기 위해 별도 평가셋을 설계했습니다.

### Evaluation Categories
- **Single-document extraction**
- **Single-document deep lookup**
- **Multi-document comparison**
- **Follow-up QA**
- **Abstention / no-evidence detection**

### 주요 지표
- **Answer Accuracy**
- **Groundedness Rate**
- **Citation Precision**
- **Abstention Accuracy**
- **Follow-up Consistency**
- **Latency (p50 / p95)**
- **Retry Rate**

### 왜 이 지표를 썼는가
RFP QA에서는 “말이 되는 답변”보다 아래가 중요합니다.

- 문서 근거가 있는가
- 비교 질문에 양쪽 문서를 모두 반영했는가
- 문서에 없는 내용은 추측하지 않는가
- 후속 질문에서도 맥락이 유지되는가

---

## 10. 베이스라인과 개선 과정

### Baseline
**Dense retrieval + LLM answer generation**

### Improvement 1 — Metadata Filtering
- 기관명 / 사업명 / 공고 정보 기반 후보 문서 축소
- 불필요한 문서가 retrieval 상위에 뜨는 문제 완화

### Improvement 2 — Query Analyzer + Planner
- 질문 유형별 검색 전략 분기
- 비교형 / 후속질문형 / 부재판단형 처리 안정화

### Improvement 3 — Reranking + Evidence Aggregation
- 관련 청크를 재정렬하고, 최종 답변에 필요한 근거를 통합

### Improvement 4 — Verifier + Retry Loop
- 근거 부족 답변 억제
- retrieval 실패 시 재검색
- groundedness 개선

---

## 11. 예시 질의

### Example 1
**Q. 국민연금공단이 발주한 이러닝 시스템 관련 사업 요구사항을 정리해줘.**

**A.**
- 사업 목적: ...
- 주요 요구사항: ...
- 제출 관련 정보: ...
- 근거 문서: `[문서명]`, p.xx

### Example 2
**Q. 고려대학교 차세대 포털 시스템 사업과 광주과학기술원 학사 시스템 기능개선 사업을 비교해줘.**

**A.**
| 항목 | 고려대학교 | 광주과학기술원 |
|---|---|---|
| 사업 목적 | ... | ... |
| 주요 기능 | ... | ... |
| 성능 요구 | ... | ... |
| 응답 시간 요구 | ... | ... |

### Example 3
**Q. 이 문서에서 AI 기반 예측 요구사항이 있나?**

**A.**
명시적으로 확인되는 AI 기반 예측 요구사항은 없습니다.  
관련 키워드 및 유사 표현으로 재검색했으나 충분한 근거를 찾지 못했습니다.

---

## 12. 결과 표 템플릿

> 아래 표는 실제 실험 결과로 교체해 사용할 수 있도록 남겨둔 템플릿입니다.

| Setting | Accuracy | Groundedness | Abstention | Latency |
|---|---:|---:|---:|---:|
| Baseline Dense RAG | `[xx.x]` | `[xx.x]` | `[xx.x]` | `[x.xx s]` |
| + Metadata Filter | `[xx.x]` | `[xx.x]` | `[xx.x]` | `[x.xx s]` |
| + Planner/Reranker | `[xx.x]` | `[xx.x]` | `[xx.x]` | `[x.xx s]` |
| Final Agentic RAG | `[xx.x]` | `[xx.x]` | `[xx.x]` | `[x.xx s]` |

### Key Observations
- 메타데이터 필터링만 추가해도 irrelevant chunk가 감소할 가능성이 큽니다.
- 비교형 질문은 planner + reranking 도입 효과가 크게 나타날 수 있습니다.
- verifier 추가 후 groundedness가 개선되고 hallucination이 감소하는 경향을 기대할 수 있습니다.
- 후속 질문 처리에서는 context 유지 전략의 품질 차이가 분명하게 드러납니다.

---

## 13. 실패 사례 분석

### 13.1 기관명 표현 불일치
- 문제: 사용자 질문과 메타데이터 표기가 달라 exact match 실패
- 대응: alias / 유사도 기반 보정

### 13.2 비교 질문 편향
- 문제: 비교형 질문에서 한쪽 문서만 retrieval
- 대응: planner 단계에서 비교 대상을 먼저 추출하고 문서별 검색 분리

### 13.3 근거 없는 일반화
- 문제: 생성 모델이 문서 바깥 일반지식으로 답변
- 대응: abstention rule + verifier 강화

### 13.4 긴 문서의 핵심 요구사항 누락
- 문제: 청킹 단위에 따라 중요한 표/문맥이 분리됨
- 대응: overlap 조정 및 chunk aggregation 실험

---

## 14. 저장소 구조 예시

```text
.
├── app
│   ├── ingestion
│   │   ├── pdf_loader.py
│   │   ├── hwp_loader.py
│   │   └── metadata_parser.py
│   ├── indexing
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   └── vector_store.py
│   ├── agent
│   │   ├── query_analyzer.py
│   │   ├── planner.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   ├── answerer.py
│   │   └── verifier.py
│   ├── eval
│   │   ├── eval_dataset.jsonl
│   │   ├── metrics.py
│   │   └── run_eval.py
│   └── ui
│       └── demo_app.py
├── experiments
├── logs
├── docs
├── reports
├── requirements.txt
└── README.md
```

> 위 구조는 예시입니다. 현재 저장소 구조에 맞게 디렉토리명만 조정하면 됩니다.

---

## 15. 실행 방법

### 15.1 설치
```bash
git clone [REPO_URL]
cd [REPO_NAME]
pip install -r requirements.txt
```

### 15.2 환경 변수 설정
`.env` 파일에 필요한 키를 설정합니다.

```bash
OPENAI_API_KEY=...
# 기타 필요한 환경변수
```

### 15.3 인덱싱
```bash
python -m app.indexing.build_index
```

### 15.4 실행
```bash
python -m app.ui.demo_app
```

또는

```bash
uvicorn app.main:app --reload
```

> 위 명령어는 예시입니다. 현재 저장소 구조에 맞는 실제 실행 명령으로 바꿔주세요.

---

## 16. 이 프로젝트가 보여주는 역량

이 프로젝트를 통해 다음 역량을 검증할 수 있습니다.

- **Document AI / RAG 시스템 설계 역량**
- **비정형 문서 처리 및 메타데이터 활용 능력**
- **검색 품질 개선을 위한 retrieval engineering**
- **근거 기반 응답 설계와 hallucination 억제**
- **비교형/후속질문형 QA 처리**
- **평가셋 설계 및 실험 기반 개선**
- **1인 개발로 end-to-end ownership 수행**

---

## 17. 내가 수행한 역할

본 프로젝트는 1인 프로젝트로 진행했으며, 아래 영역을 전부 직접 수행했습니다.

- 문제 정의 및 범위 설정
- 데이터 구조 파악 및 문서 처리 파이프라인 설계
- 청킹/임베딩/벡터DB 구축
- query analyzer / planner / retriever / verifier 설계 및 구현
- 평가셋 설계 및 지표 정의
- 실험 로그 분석 및 성능 개선
- README / 보고서 / 발표 자료 작성

---

## 18. 한계

- HWP 파싱 품질은 문서 구조에 따라 편차가 있습니다.
- 표/도식 중심 정보는 텍스트 기반 접근만으로 한계가 있습니다.
- 비교형 질문의 정합성 검증은 여전히 어려운 케이스가 존재합니다.
- 장문 멀티턴 대화에서는 context 관리 최적화 여지가 있습니다.

---

## 19. 향후 개선 방향

- 표/이미지까지 다루는 멀티모달 문서 이해 확장
- domain-specific reranker 적용
- retrieval trace 시각화
- structured extraction cache 도입
- 질의 유형별 adaptive planning 강화

---

## 20. 회고

이 프로젝트를 통해 단순한 RAG 데모를 넘어서,  
**문서 검색 시스템이 실제 의사결정 보조 도구가 되려면 무엇이 필요한지**를 구체적으로 고민하고 구현할 수 있었습니다.

특히 다음을 중요하게 느꼈습니다.

- 좋은 QA 시스템은 좋은 생성보다 **좋은 retrieval과 evidence structure** 에 더 크게 의존합니다.
- 비교형 질문과 후속 질문은 단순 top-k retrieval만으로 해결되지 않습니다.
- “모른다”고 정확히 답하는 능력이 실무형 시스템에서는 오히려 핵심입니다.
- 로그와 평가셋이 없으면 개선도, 설득도 어렵습니다.

---

## 21. 관련 문서

- Report: `[링크 추가]`
- Presentation: `[링크 추가]`
- Dev Log / Journal: `[링크 추가]`

---

## 22. Notice

본 저장소는 프로젝트 코드, 실험 구조, 평가 결과, 2차 가공 산출물만 포함합니다.  
원본 데이터(RFP 문서)는 비공개이며 외부 공유가 제한됩니다.

