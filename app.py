import streamlit as st
from process_fasta import build_vectorstore, inspect_vectorstore

# 세션 상태 초기화
if "running" not in st.session_state:
    st.session_state.running = False

if "stop_requested" not in st.session_state:
    st.session_state.stop_requested = False

st.set_page_config(page_title="PDB VectorStore Builder", layout="centered")
st.title("🔬 PDB VectorStore 생성기")
st.markdown("RCSB에서 단백질 정보를 수집하고 FAISS 벡터스토어로 저장합니다.")

# 버튼 1: 벡터스토어 생성 시작
if st.button("🚀 벡터스토어 생성 시작"):
    st.session_state.running = True
    st.session_state.stop_requested = False

# 버튼 2: 일시 정지 버튼
if st.session_state.running and st.button("⏸️ 일시 정지"):
    st.session_state.stop_requested = True

# 버튼 3: 벡터스토어 정보 확인
if st.button("🔍 벡터스토어 정보 확인"):
    st.info("벡터스토어를 분석 중입니다...")
    inspect_vectorstore(save_path="./vector_db/fasta")


# 벡터스토어 생성 실행
if st.session_state.running:
    st.info("벡터스토어를 생성 중입니다. 잠시만 기다려 주세요...")

    # Streamlit UI 요소 준비
    log_area = st.empty()
    progress_bar = st.progress(0)

    # 로그 출력 함수 정의
    def log_fn(msg):
        log_area.text(msg)

    def should_continue():
        return not st.session_state.stop_requested

    # 벡터스토어 생성 시작
    build_vectorstore(
        batch_size=100,
        total=1000,
        save_path="./vector_db/fasta",
        log_fn=log_fn,
        progress_bar=progress_bar,
        should_continue=should_continue
    )

    if st.session_state.stop_requested:
        st.warning("⏹️ 중단되었습니다.")
    else:
        st.success("✅ 벡터스토어 저장이 완료되었습니다!")

    st.session_state.running = False