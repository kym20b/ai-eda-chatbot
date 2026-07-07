"""
AI 노코드 데이터 분석 조수 (EDA 챗봇)
────────────────────────────────────────────────────────────
실행 전 준비
  1. pip install streamlit pandas plotly google-generativeai
  2. 프로젝트 루트에 .streamlit/secrets.toml 생성 후 아래 내용 추가:
       GEMINI_API_KEY = "your-gemini-api-key-here"
  3. streamlit run app.py
────────────────────────────────────────────────────────────
"""

import io
import re
import pandas as pd
import plotly.express as px
import streamlit as st
import google.generativeai as genai


# ════════════════════════════════════════════════════════════════════════════
# 1. 페이지 기본 설정
#    - layout="wide" : 좌우 여백 없이 넓게 사용
#    - page_icon, page_title : 브라우저 탭에 표시
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI 노코드 데이터 분석 조수",
    page_icon="🤖",
    layout="wide",
)


# ════════════════════════════════════════════════════════════════════════════
# 2. Gemini API 키 로드 및 모델 초기화
#    - API 키는 절대 하드코딩 금지. st.secrets 를 통해 안전하게 불러옴.
#    - .streamlit/secrets.toml → GEMINI_API_KEY = "sk-..."
#    - 키가 없으면 에러 표시 후 앱 중단 (st.stop)
# ════════════════════════════════════════════════════════════════════════════
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    # Gemini 3.1 Flash Lite: 빠른 응답 속도 + 긴 컨텍스트를 지원하는 경량 모델
    gemini_model = genai.GenerativeModel("gemini-3.1-flash-lite")
except KeyError:
    st.error(
        "⚠️ **GEMINI_API_KEY 가 설정되어 있지 않습니다.**\n\n"
        "프로젝트 루트에 `.streamlit/secrets.toml` 파일을 만들고 아래 내용을 추가하세요:\n\n"
        "```toml\n"
        'GEMINI_API_KEY = "your-gemini-api-key-here"\n'
        "```\n\n"
        "Gemini API 키는 [Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급받을 수 있습니다."
    )
    st.stop()  # 키 없이는 앱 실행 불가 → 여기서 스크립트 중단


# ════════════════════════════════════════════════════════════════════════════
# 3. 세션 상태 초기화
#    st.session_state 는 Streamlit 이 페이지를 리렌더링해도 값을 유지하는 저장소.
#    앱 최초 로드 시 딱 한 번만 초기화되도록 "in" 으로 존재 여부를 체크.
#
#    messages 리스트의 각 항목 구조:
#    {
#      "role":    "user" | "assistant",  ← 발화자
#      "content": str | None,            ← 텍스트 메시지
#      "code":    str | None,            ← AI 가 생성한 Python 코드
#      "fig":     Figure | None,         ← Plotly 차트 객체
#      "result":  any | None,            ← 차트 외 분석 결과 (DataFrame, 숫자 등)
#      "report":  str | None,            ← 차트 기반 AI 분석 리포트 (fig 있을 때만)
#      "error":   str | None,            ← 에러 메시지
#    }
# ════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state:
    st.session_state.messages = []


# ════════════════════════════════════════════════════════════════════════════
# 4. 사이드바 ─ CSV 업로드 & 데이터 메타데이터 추출
# ════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("📂 데이터 업로드")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        label="CSV 파일을 선택하거나 드래그하세요",
        type=["csv"],
        help="UTF-8 인코딩의 CSV 파일을 권장합니다.",
    )

    # df 와 df_info_str 은 이후 메인 화면에서도 사용하므로 None 으로 초기화
    df: pd.DataFrame | None = None
    df_info_str: str = ""

    if uploaded_file is not None:

        # ── 4-1. CSV 파싱
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as read_err:
            st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다:\n`{read_err}`")
            st.stop()

        st.success(f"✅ 로드 완료: **{df.shape[0]:,}행 × {df.shape[1]}열**")

        # ── 4-2. 상위 5행 미리보기
        st.markdown("**📋 데이터 미리보기 (상위 5행)**")
        st.dataframe(df.head(), use_container_width=True)

        # ── 4-3. df.info() 결과를 문자열로 캡처
        #         df.info() 는 기본적으로 stdout 에 출력하므로
        #         io.StringIO 버퍼를 buf= 인자로 넘겨 문자열로 리다이렉트
        info_buf = io.StringIO()
        df.info(buf=info_buf)
        df_info_str = info_buf.getvalue()  # LLM 프롬프트에 삽입할 문자열

        with st.expander("🔍 데이터 구조 (df.info)", expanded=False):
            st.text(df_info_str)

        with st.expander("📊 기술 통계 (df.describe)", expanded=False):
            st.dataframe(df.describe(), use_container_width=True)

        st.markdown("---")

        # ── 4-4. 대화 초기화 버튼 (새 파일 분석 시 이전 대화 정리용)
        if st.button("🔄 대화 기록 초기화", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("Powered by Gemini 3.1 Flash Lite · Built with Streamlit")


# ════════════════════════════════════════════════════════════════════════════
# 5. 메인 화면 ─ 타이틀 & 설명
# ════════════════════════════════════════════════════════════════════════════
st.title("🤖 AI 노코드 데이터 분석 조수")
st.markdown(
    "CSV 파일을 업로드한 뒤 분석 요청을 자연어로 입력하면, "
    "AI가 **Python 코드를 생성하고 즉시 실행**해 차트를 그려드립니다."
)
st.markdown("---")


# ════════════════════════════════════════════════════════════════════════════
# 6. 대화 이력 렌더링 헬퍼 함수
#    세션 상태에 저장된 메시지 딕셔너리 하나를 받아 화면에 출력.
#    st.chat_message 컨텍스트 안에서 호출해야 올바르게 렌더링됨.
# ════════════════════════════════════════════════════════════════════════════
def render_message(msg: dict) -> None:
    """세션에 저장된 메시지를 종류에 맞게 화면에 렌더링한다."""

    # 일반 텍스트
    if msg.get("content"):
        st.markdown(msg["content"])

    # AI 가 생성한 Python 코드 블록
    if msg.get("code"):
        st.markdown("**🧑‍💻 AI가 생성한 코드**")
        st.code(msg["code"], language="python")

    # Plotly 인터랙티브 차트
    if msg.get("fig") is not None:
        st.plotly_chart(msg["fig"], use_container_width=True)

    # 차트 외 분석 결과 (DataFrame, 숫자, 문자열 등)
    if msg.get("result") is not None:
        st.markdown("**📊 분석 결과**")
        result = msg["result"]
        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True)
        else:
            st.write(result)

    # AI 분석 리포트 (차트 아래에 표시)
    if msg.get("report"):
        st.markdown("---")
        st.markdown("**📋 AI 분석 리포트**")
        st.info(msg["report"])

    # 에러 메시지
    if msg.get("error"):
        st.error(msg["error"])


# ── 세션에 쌓인 대화 이력을 위에서부터 순서대로 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        render_message(msg)


# ════════════════════════════════════════════════════════════════════════════
# 7. 챗 입력창 및 AI 응답 파이프라인
#    파일이 업로드되지 않은 경우: 안내 메시지 + 비활성 입력창
#    파일이 업로드된 경우: 입력창 활성화 → 질문 수신 → AI 처리
# ════════════════════════════════════════════════════════════════════════════
if df is None:
    # 파일 미업로드 상태 안내
    st.info("👈 왼쪽 사이드바에서 **CSV 파일을 먼저 업로드**해 주세요!")
    st.chat_input(
        placeholder="CSV 파일을 업로드해야 대화를 시작할 수 있습니다.",
        disabled=True,
    )

else:
    # ── 7-1. 사용자 입력 수신
    user_input: str | None = st.chat_input(
        placeholder="예) '연도별 평균 매출을 막대 차트로 그려줘', '결측값 개수를 알려줘'"
    )

    if user_input:

        # ── 7-2. 사용자 메시지 저장 & 즉시 렌더링
        user_msg = {"role": "user", "content": user_input, "code": None,
                    "fig": None, "result": None, "report": None, "error": None}
        st.session_state.messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── 7-3. Gemini 에게 보낼 프롬프트 구성
        #         ① 역할 지정  ② 데이터 구조 주입  ③ 사용자 요구  ④ 코드 작성 규칙
        prompt = f"""
너는 숙련된 데이터 분석가이자 Python 전문가다.

[현재 실행 환경]
- 사용자가 업로드한 데이터는 `df` 라는 Pandas DataFrame 변수에 이미 저장되어 있다.
- `plotly.express` 는 `px` 라는 이름으로 이미 임포트되어 있다.
- `pandas` 는 `pd` 라는 이름으로 이미 임포트되어 있다.
- 따라서 코드에 import 문을 별도로 작성하지 마라.

[데이터 구조 정보 (df.info 결과)]
{df_info_str}

[사용자 요구사항]
{user_input}

[코드 작성 규칙 — 반드시 모두 준수]
1. 데이터 접근은 반드시 `df` 변수를 사용해라.
2. 시각화가 필요한 경우 반드시 `plotly.express` (px) 를 사용해라.
3. 최종 Plotly Figure 객체는 반드시 `fig` 변수에 저장해라.
4. 시각화 없이 통계·집계 결과만 반환하는 경우, 결과를 `result` 변수에 저장해라.
5. 코드 설명, 주석, 마크다운 텍스트는 절대 포함하지 마라.
6. 반드시 아래 형식의 마크다운 코드 블록 하나만 반환해라:

```python
# 여기에 코드 작성
```
""".strip()

        # ── 7-4. Gemini API 호출 → 코드 추출 → 실행 → 결과 렌더링
        with st.chat_message("assistant"):
            with st.spinner("🤖 AI가 코드를 생성 중입니다..."):

                # ── [STEP A] Gemini API 호출
                api_call_ok = True
                raw_text = ""
                try:
                    response = gemini_model.generate_content(prompt)
                    raw_text = response.text
                except Exception as api_err:
                    err_msg = (
                        f"🚨 **Gemini API 호출 실패**\n\n"
                        f"`{type(api_err).__name__}: {api_err}`"
                    )
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant", "content": None, "code": None,
                        "fig": None, "result": None, "report": None, "error": err_msg,
                    })
                    api_call_ok = False

            # API 실패 시 이하 처리 건너뜀
            if api_call_ok:

                # ── [STEP B] 코드 블록 추출 (정제/Sanitization)
                #    Gemini 는 ```python ... ``` 형식으로 반환하므로
                #    정규식으로 코드 블록 내부의 순수 코드만 분리
                code_match = re.search(
                    r"```(?:python)?\s*([\s\S]*?)```",
                    raw_text,
                    re.IGNORECASE,
                )
                if code_match:
                    clean_code = code_match.group(1).strip()
                else:
                    # 코드 블록 마커가 없는 예외적 상황: 백틱만 제거하고 사용
                    clean_code = raw_text.replace("```", "").strip()

                # ── [STEP C] 수강생에게 생성된 코드 공개 (교육 목적)
                st.markdown("**🧑‍💻 AI가 생성한 코드**")
                st.code(clean_code, language="python")

                # ── [STEP D] 코드 동적 실행 (exec)
                #    - globals 에 __builtins__ 를 포함시켜 기본 내장함수 사용 가능
                #    - locals (local_ns) 에 df, px, pd 를 미리 바인딩
                #    - 실행 후 local_ns 에서 fig / result 를 꺼내 렌더링
                local_ns: dict = {
                    "df": df,   # 사용자가 업로드한 DataFrame
                    "px": px,   # plotly.express
                    "pd": pd,   # pandas
                }

                try:
                    exec(
                        clean_code,
                        {"__builtins__": __builtins__},  # 안전한 글로벌 네임스페이스
                        local_ns,                         # df/px/pd + 실행 결과 변수 수집
                    )

                    fig    = local_ns.get("fig",    None)
                    result = local_ns.get("result", None)

                    # ── 결과 종류에 따라 분기 렌더링
                    if fig is not None:
                        # Plotly 인터랙티브 차트 출력
                        st.plotly_chart(fig, use_container_width=True)

                        # ── [STEP D-2] 차트 기반 AI 분석 리포트 생성
                        #    df.describe() 를 문자열로 변환해 프롬프트에 삽입
                        report_text = None
                        try:
                            describe_str = df.describe(include="all").to_string()
                            report_prompt = f"""
너는 데이터 분석 전문가다. 아래 정보를 바탕으로 차트에서 읽을 수 있는 핵심 인사이트를 짧은 리포트로 작성해라.

[사용자 질문]
{user_input}

[실행된 분석 코드]
{clean_code}

[데이터 기본 통계 (df.describe)]
{describe_str}

[작성 규칙]
- 3~5개의 핵심 인사이트를 불릿 포인트(•)로 작성해라.
- 각 항목은 1~2문장으로 간결하게 작성해라.
- 구체적인 수치나 비율을 포함해 설득력 있게 써라.
- 마크다운 볼드(**텍스트**)를 활용해 핵심 수치를 강조해라.
- 반드시 한국어로 작성해라.
- 코드나 코드 블록은 절대 포함하지 마라.
""".strip()
                            with st.spinner("📋 분석 리포트를 작성 중입니다..."):
                                report_resp = gemini_model.generate_content(report_prompt)
                                report_text = report_resp.text.strip()

                            st.markdown("---")
                            st.markdown("**📋 AI 분석 리포트**")
                            st.info(report_text)

                        except Exception:
                            # 리포트 생성 실패 시 차트 결과에는 영향 없이 조용히 넘어감
                            pass

                        assistant_msg = {
                            "role": "assistant",
                            "content": "✅ 차트가 생성되었습니다!",
                            "code": clean_code,
                            "fig": fig,
                            "result": None,
                            "report": report_text,
                            "error": None,
                        }

                    elif result is not None:
                        # 차트 없는 분석 결과 출력 (DataFrame, 숫자 등)
                        st.markdown("**📊 분석 결과**")
                        if isinstance(result, pd.DataFrame):
                            st.dataframe(result, use_container_width=True)
                        else:
                            st.write(result)
                        assistant_msg = {
                            "role": "assistant",
                            "content": "✅ 분석 결과입니다.",
                            "code": clean_code,
                            "fig": None,
                            "result": result,
                            "report": None,
                            "error": None,
                        }

                    else:
                        # fig 도 result 도 없는 경우: AI 가 다른 변수명을 사용했을 가능성
                        warn_msg = (
                            "⚠️ 코드는 실행됐지만 `fig` 또는 `result` 변수를 찾지 못했습니다. "
                            "질문을 더 구체적으로 수정하거나 다시 시도해 보세요."
                        )
                        st.warning(warn_msg)
                        assistant_msg = {
                            "role": "assistant",
                            "content": warn_msg,
                            "code": clean_code,
                            "fig": None,
                            "result": None,
                            "report": None,
                            "error": None,
                        }

                except Exception as exec_err:
                    # 코드 실행 중 런타임 에러 → 앱이 뻗지 않도록 부드럽게 출력
                    err_msg = (
                        f"🚨 **코드 실행 중 오류가 발생했습니다**\n\n"
                        f"`{type(exec_err).__name__}: {exec_err}`\n\n"
                        "질문을 다르게 표현하거나 컬럼명을 구체적으로 지정해 보세요."
                    )
                    st.error(err_msg)
                    assistant_msg = {
                        "role": "assistant",
                        "content": None,
                        "code": clean_code,
                        "fig": None,
                        "result": None,
                        "report": None,
                        "error": err_msg,
                    }

                # ── [STEP E] 어시스턴트 메시지를 세션에 저장
                #    다음 리렌더링 시 render_message() 가 이 데이터를 복원해 표시함
                st.session_state.messages.append(assistant_msg)
