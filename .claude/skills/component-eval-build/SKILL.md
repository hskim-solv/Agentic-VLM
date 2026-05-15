---
name: component-eval-build
description: |
  Build component-level evaluation infrastructure for the BidMate-DocAgent agentic RAG pipeline (Query Analyzer → Planner → Retriever → Verifier/Retry → Answer Generator). Drives a strict 5-phase workflow — eval-set diagnosis & rebuild → component isolation harness → process/trajectory metrics → statistical rigor → closed error loop — with hard rules against fake metrics, documentation-before-code drift, premature abstraction, and silent retcon.

  Trigger phrases (English): "build component eval", "component-level eval harness", "phased eval infrastructure", "set up eval infra", "/component-eval-build".
  Trigger phrases (Korean): "컴포넌트 평가 인프라 만들자", "phased eval 작업 시작", "eval harness 구축", "이거 phase별로 진행하자". Also trigger when the user references "Phase 1 step 1" / "다음 phase 진행" within an already-active session of this skill.

  Do NOT trigger for: ad-hoc single-metric measurement, one-off retrieval debugging, eval config tweaks that don't touch the harness, reading an existing eval result, generic RFP DocAgent work unrelated to evaluation infrastructure. Trigger ONLY when the user wants to BUILD or EXTEND the component-level harness itself.
---

# /component-eval-build — Component-level eval infrastructure builder

## Context

`BidMate-DocAgent`는 RFP 문서 분석용 agentic RAG 시스템.
현재 pipeline: Query Analyzer → Planner → Retriever → Verifier/Retry → Answer Generator.

**현재 진단된 약점:**
- Eval set이 작아 통계적 검정력이 부족함
- Ablation 결과가 flat — eval set의 distinguishing power 부족 또는 component isolation 부재가 의심됨
- "Agentic" framing이 모호함 — 사실상 conditional pipeline일 가능성 있음
- 과거에 documentation이 implementation보다 앞서 나가는 패턴이 반복됨 — **이번에는 절대 그러지 말 것**

## Goal

다음을 수행할 수 있는 component-level evaluation infrastructure를 구축한다.

1. 각 component를 oracle input으로 격리 측정
2. Oracle ceiling 산출 (counterfactual: "이 component가 완벽하면 천장은 어디인가")
3. Noise와 진짜 개선을 통계적으로 구분
4. Process metric (latency, retry count, $/query) 수집
5. Final answer 정확도뿐 아니라 trajectory rationality 측정
6. Failure taxonomy를 closed loop로 운영 (실패 → eval set 성장)

## 절대 규칙 (먼저 읽을 것)

- **코드 먼저, 문서 마지막.** 해당 코드가 실제 데이터에서 동작해 non-trivial output을 내기 전에는 README, 아키텍처 다이어그램, 장문의 docstring 금지. 코드 없이 markdown부터 쓰고 있다면 멈출 것.
- **Fake metric 금지.** 모든 숫자는 실제 데이터에서 실제 실행한 결과여야 하며 재현 커맨드가 기록되어야 함. Latency 추정치, "expected" recall@k 같은 거 만들지 말 것. 측정되지 않은 metric은 존재하지 않는 것.
- **조기 추상화 금지.** 각 component의 가장 단순한 동작 버전부터 만들고, 구체적 use case가 2개 이상 생기기 전에는 base class / interface 설계 금지.
- **새 의존성 추가 시 정당화 필수.** stdlib + pytest + pandas + 기존 deps 우선. 라이브러리 추가 시 기존 도구로 안 되는 이유 한 줄 작성.
- **실패 정직 보고.** 어떤 phase에서 이전 주장이 틀렸음이 드러나면 phase report에 명시적으로 보고할 것. 조용히 retcon 금지.

## Phased plan

순서대로 진행. 각 phase 종료 시 **STOP**하고 phase report (markdown, 200줄 이내) 작성: 무엇을 만들었는지, 무엇을 측정했는지 (raw 숫자), 무엇이 예상과 달랐는지, 다음 phase의 위험은 무엇인지. 사람 리뷰를 받기 전 다음 phase 진행 금지.

### Phase 1 — Eval set 진단 + 재구축

1. 현재 eval set 로드. 출력: n, query 길이 분포, gold answer 길이 분포, 카테고리 분포(있다면), no-answer 비율.
2. **Distinguishing-power test.** 의도적으로 망가뜨린 baseline 3개 정의:
   - `random_retrieval`: retriever를 random chunk sampling으로 교체
   - `no_verifier`: verifier/retry loop 제거
   - `single_chunk`: top-1 chunk만 사용, context assembly 없음

   3개 + 현재 production pipeline을 기존 eval set에서 실행. End-to-end accuracy 보고. 망가진 baseline 중 어느 하나라도 production과 5%p 이내라면 → **eval set이 문제**. 보고서에 prominently flag.
3. **Eval set을 n ≥ 200으로 확장**, 다음 카테고리별 명시적 균형:
   - Single-hop factual
   - Multi-hop (2개 이상 chunk 조합 필요)
   - Ambiguous query (복수 해석 가능)
   - No-answer (corpus에 정보 부재)
   - Distractor-heavy (그럴듯한 오답 chunk 다수)
   - Long-context (관련 정보가 5k+ token 깊이에 위치)

   카테고리당 ≥ 30개 목표. 카테고리별 생성 방법 문서화.
4. **Layered gold annotations.** 각 예제마다 기록:
   - Gold final answer
   - Gold sub-queries (planner 평가 기준)
   - Gold evidence chunk IDs (retriever 평가 기준)
   - Gold reasoning summary (trajectory 평가 기준)
   - 난이도 라벨 (easy/medium/hard)
5. 새 eval set에서 3개 broken baseline 재실행. 각각이 production 대비 ≥ 10%p 낮음을 확인. 아니면 → 아직 distinguishing power 부족, 다시 보강.

**Phase 1 acceptance:** broken baseline이 새 eval set에서 production과 명확히 분리됨; 카테고리별 size 문서화됨; layered annotation이 machine-readable (jsonl/parquet).

### Phase 2 — Component isolation harness

1. 각 component를 oracle-injected upstream input으로 평가할 수 있는 인터페이스 정의:
   - `planner_eval(query) → sub_queries`: gold sub-queries 대비 decomposition F1 + LLM-as-judge rubric
   - `retriever_eval(sub_query OR oracle_sub_queries) → chunks`: gold chunk ID 대비 recall@k, MRR, nDCG
   - `verifier_eval(context, claim) → sufficient?`: 라벨링된 (context, claim, sufficiency) 100쌍 이상 (양성/음성 균형)에서 precision/recall
   - `generator_eval(query, oracle_context) → answer`: faithfulness, answer correctness, citation accuracy
2. **Oracle injection switch**를 각 pipeline 경계에 구현. config flag 하나로 real vs oracle input 토글 가능해야 함.
3. End-to-end eval을 5개 조건에서 실행: 전부 real / oracle planner / oracle retriever / oracle verifier / oracle generator. 각 조건의 delta가 해당 component의 ceiling 기여도를 드러냄.

**Phase 2 acceptance:** 5개 조건 모두 CLI 한 줄로 실행 가능; 각 component metric의 raw 숫자 + 샘플 단위 결과 저장.

### Phase 3 — Process + trajectory metrics

1. Pipeline에 per-query 로깅 추가: total latency, component별 latency, retriever call count, verifier retry count, token in/out, 추정 $.
2. 전체 trajectory (모든 LLM call의 input/output) 구조화 포맷으로 eval 예제별 저장.
3. Trajectory-rationality rubric 정의 (LLM-as-judge): "Planner decomposition이 query 의도와 일치하는가? Retrieval 재호출이 합당한 이유로 발생했는가? Verifier 판정이 evidence와 정합적인가?" 차원별 1-5 점.
4. **Pareto reporting:** 모든 system variant 비교는 quality vs cost를 2D plot으로 보고. Latency 3배 늘려서 accuracy 2% 얻은 변형은 대부분 context에서 regression임.

**Phase 3 acceptance:** trajectory 직렬화 완료; per-query cost/latency 조회 가능; trajectory rationality judge가 전체 eval set에서 무사고 실행.

### Phase 4 — 통계적 엄밀성

1. 각 system variant를 **3 seed**로 실행, mean ± std 보고.
2. Variant 비교 시 동일 eval 예제에 대한 **paired bootstrap CI** 사용. Δ accuracy를 95% CI와 함께 보고.
3. `claim_validator.py` 추가: 개선 주장(e.g. "verifier가 accuracy를 4% 개선")을 입력받아 측정된 Δ, CI, sample size, p-value 출력. CI가 0을 가로지르는 주장은 명시적 "NOT SIGNIFICANT" 태그 없이는 출력 거부.

**Phase 4 acceptance:** 모든 report의 모든 metric이 CI 동반; "X improves Y" 주장은 자동 검증을 통과해야만 보고됨.

### Phase 5 — Closed error loop

1. Failure taxonomy 정의 (시작: retrieval_miss / planner_under_decomposition / verifier_false_negative / verifier_false_positive / generator_hallucination / context_dilution / unknown). 모든 오답에 태그.
2. Failure distribution 대시보드 생성 (bar chart + table).
3. Process: 기존 카테고리에 안 맞는 새 failure mode 발견 시 → 카테고리 추가 + 해당 패턴 예제 ≥ 5개를 eval set에 추가. Eval set이 시간이 갈수록 monotonically harder해질 것.

**Phase 5 acceptance:** failure distribution 산출 가능; 이번 phase 중 실제로 새 카테고리 1개 추가 (조작 금지, 실제 데이터에서 발견할 것).

## Repo conventions

- `evals/`: eval set 데이터 및 스크립트. `evals/run.py`가 entrypoint
- `metrics/`: metric 구현 (모듈별 분리 + unit test)
- `reports/`: phase report 및 결과 스냅샷 (jsonl + markdown summary)
- `pytest` 사용. metric monotonicity 같은 곳엔 property-based test 적용
- 모든 eval 실행은 저장: timestamp, git commit hash, config, per-example raw 결과, aggregated metrics
- Random seed 명시적; 숨은 nondeterminism 금지

## 턴 단위 작업 방식

- 매 턴: 1개 phase 분량 (또는 blocker 만나면 그 이하)
- 매 턴 종료 시 보고: 만든 파일 목록 + LOC, 측정한 숫자, 예상과 달랐던 점, 우려되는 점
- 사용자 승인 없이 다음 phase 시작 금지
- 이전 phase의 주장이 틀렸음이 드러나면 다음 report에 명시. 조용한 retcon 금지

## 첫 번째 action

Phase 1의 step 1만 수행: 현재 eval set 로드하고 프로파일 출력. 새 코드 작성 금지. 지금 무엇이 있는지만 보고할 것.
