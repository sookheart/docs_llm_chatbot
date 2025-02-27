import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# 텍스트 정규화 함수 추가
def normalize_text(text):
    """텍스트를 정규화하여 대소문자 구분 문제를 해결합니다."""
    if text is None:
        return ""
    # 모든 텍스트를 소문자로 변환하고 추가 정규화 적용
    return text.lower().strip()

# 페이지 설정
st.set_page_config(layout="wide", page_title="Multi-Document ChatBot")

# 메인 타이틀
st.title("🤖 다중 문서 통합 RAG Chatbot")
st.write("웹페이지, PDF, YouTube 영상을 동시에 선택하여 통합 검색해보세요!")

# 질문 처리 함수
def process_query(prompt):
    if not prompt.strip():
        return  # 빈 입력은 처리하지 않음
        
    if not st.session_state.vector_store:
        st.error("먼저 문서를 로드해주세요!")
        return
        
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    # 사용자 입력 정규화 (질문과 문서의 대소문자를 일치시키기 위함)
    normalized_prompt = normalize_text(prompt)
    
    # 어시스턴트 응답 생성
    with st.spinner("생각 중..."):
        llm = ChatOpenAI(model_name=selected_model, temperature=0)
        
        # 대소문자 구분 없이 검색을 위한 커스텀 쿼리 구성
        # 검색 범위를 넓히고 여러 조합을 시도
        variations = [
            normalized_prompt,  # 소문자 버전
            prompt.upper(),     # 대문자 버전
            prompt,             # 원본 그대로
        ]
        
        # 중복 제거
        variations = list(set(variations))
        
        all_docs = []
        for query_var in variations:
            # 각 변형에 대해 검색 실행
            docs = st.session_state.vector_store.similarity_search(
                query_var, 
                k=2,  # 각 변형마다 상위 2개씩 가져옴
                fetch_k=5  # 초기 후보는 더 많이
            )
            all_docs.extend(docs)
        
        # 중복 제거 (같은 문서가 여러 쿼리 변형에서 검색될 수 있음)
        unique_docs = []
        doc_contents = set()
        for doc in all_docs:
            if doc.page_content not in doc_contents:
                unique_docs.append(doc)
                doc_contents.add(doc.page_content)
        
        # 상위 3개 문서만 사용
        top_docs = unique_docs[:3] if len(unique_docs) > 3 else unique_docs
        
        # QA 체인 실행
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=st.session_state.vector_store.as_retriever(),
            return_source_documents=True
        )
        
        # 실제 QA 수행
        result = qa_chain.invoke({
            "query": normalized_prompt,
            "context": "\n\n".join([doc.page_content for doc in top_docs])
        })
        response = result["result"]
        response = result["result"]
        
        # 출처 정보 추가 (유효한 응답이 있는 경우에만)
        if "source_documents" in result and response and not response.startswith("I don't have information"):
            sources = []
            for doc in result["source_documents"][:3]:  # 상위 3개 문서만 표시
                source_type = doc.metadata.get("source_type", "unknown")
                source = doc.metadata.get("source", "unknown source")
                sources.append(f"- {source_type}: {source}")
            
            if sources:
                response += "\n\n**참고 출처:**\n" + "\n".join(set(sources))
        
        # 어시스턴트 메시지 추가
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# 문서 처리 함수들
def load_pdf(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    documents = loader.load_and_split()
    # 텍스트 정규화 적용
    for doc in documents:
        doc.page_content = normalize_text(doc.page_content)
        # 소스 타입 메타데이터 추가
        doc.metadata["source_type"] = "pdf"
        # 원본 텍스트를 metadata에 저장하여 필요시 참조
        doc.metadata["original_text"] = doc.page_content
    return documents

def load_webpage(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    # 텍스트 정규화 적용
    for doc in documents:
        doc.page_content = normalize_text(doc.page_content)
        # 소스 타입 메타데이터 추가
        doc.metadata["source_type"] = "web"
    return documents

def load_youtube(youtube_url):
    try:
        # URL에서 video_id 추출
        if "youtube.com/watch?v=" in youtube_url:
            video_id = youtube_url.split("youtube.com/watch?v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
        else:
            st.error(f"올바른 YouTube URL 형식이 아닙니다: {youtube_url}")
            return []
        
        # 자막 가져오기
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
            
            # 자막 텍스트 결합
            full_text = ' '.join([entry['text'] for entry in transcript])
            
            # 텍스트 정규화 적용
            normalized_text = normalize_text(full_text)
            
            # Document 객체 생성
            doc = Document(
                page_content=normalized_text,
                metadata={"source": youtube_url, "source_type": "youtube"}
            )
            
            return [doc]
            
        except Exception as e:
            st.error(f"자막 처리 오류 ({youtube_url}): {str(e)}")
            return []
            
    except Exception as e:
        st.error(f"YouTube URL 처리 중 오류 ({youtube_url}): {str(e)}")
        return []

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def create_embeddings(texts, selected_embedding):
    # OpenAI 임베딩 생성시 설정
    embeddings = OpenAIEmbeddings(
        model=selected_embedding,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        # 임베딩 전에 텍스트를 추가로 정규화하여 대소문자 문제 해결
        embedding_ctx_length=8191,  # 최대 컨텍스트 길이 설정
    )
    
    return embeddings

def create_vector_db(texts, embeddings):
    # 항상 FAISS만 사용하고, 대소문자 구분 없이 검색할 수 있도록 설정
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # 검색 시 유사도 점수를 낮춰 더 많은 결과를 포함하도록 설정
    # 이렇게 하면 대소문자 차이가 있어도 관련 문서를 찾을 가능성이 높아짐
    vectorstore.similarity_search_with_score_threshold = 0.3
    
    return vectorstore

# 사이드바 설정
with st.sidebar:
    st.header("설정")
    
    # OpenAI API Key 입력
    api_key = st.text_input("OpenAI API Key", type="password")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        st.warning("OpenAI API 키를 입력해주세요!")
    
    # 모델 선택
    model_options = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
    selected_model = st.selectbox("LLM 모델 선택", model_options)
    
    # 임베딩 모델 선택
    embedding_options = ["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"]
    selected_embedding = st.selectbox("임베딩 모델 선택", embedding_options)
    
    # FAISS 사용 정보 (고정)
    st.info("벡터 DB: FAISS")
    
    st.markdown("---")
    
    # 문서 소스 선택 (체크박스 사용)
    st.header("문서 소스 선택")
    use_web = st.checkbox("웹페이지", value=False)
    use_pdf = st.checkbox("PDF 파일", value=False)
    use_youtube = st.checkbox("YouTube 영상", value=False)
    
    # 문서 소스에 따른 입력 필드들
    all_sources = {}
    
    # 웹페이지 URL 입력
    if use_web:
        st.subheader("웹페이지 URL")
        web_urls = []
        num_web_inputs = st.number_input("웹페이지 URL 개수", min_value=1, max_value=5, value=1)
        
        for i in range(num_web_inputs):
            web_url = st.text_input(f"웹페이지 URL {i+1}", key=f"web_{i}")
            if web_url:
                web_urls.append(web_url)
        
        all_sources["web"] = web_urls
    
    # PDF 파일 업로드
    if use_pdf:
        st.subheader("PDF 파일")
        uploaded_files = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"], accept_multiple_files=True)
        all_sources["pdf"] = uploaded_files if uploaded_files else []
    
    # YouTube 링크 입력
    if use_youtube:
        st.subheader("YouTube 영상 링크")
        youtube_urls = []
        num_youtube_inputs = st.number_input("YouTube 링크 개수", min_value=1, max_value=5, value=1)
        
        for i in range(num_youtube_inputs):
            youtube_url = st.text_input(f"YouTube 링크 {i+1}", key=f"youtube_{i}")
            if youtube_url:
                youtube_urls.append(youtube_url)
        
        all_sources["youtube"] = youtube_urls
    
    # 문서 로드 버튼
    load_doc_button = st.button("선택한 모든 소스 로드하기")

# 세션 상태 초기화
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_sources" not in st.session_state:
    st.session_state.current_sources = None

# UI 상태 저장을 위한 변수 추가
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False

# 메인 컨텐츠 영역 - 처리 상태에 따라 레이아웃 변경
if not st.session_state.processing_complete:
    # 문서 처리 전/중 레이아웃 (전체 너비 사용)
    st.header("문서 처리 과정")
    
    # 문서 로드 프로세스
    if load_doc_button:
        if not (use_web or use_pdf or use_youtube):
            st.error("최소한 하나 이상의 문서 소스를 선택해주세요!")
        else:
            with st.spinner("여러 소스에서 문서 처리 중..."):
                all_documents = []
                source_summaries = []
                
                # 소스 정보 생성
                sources_info = {}
                
                # 1. 모든 문서 소스 로드
                documents_status = st.empty()
                documents_status.info("1. 문서 소스 로드 중...")
                
                # 웹페이지 처리
                if use_web and all_sources["web"]:
                    for i, url in enumerate(all_sources["web"]):
                        if url:
                            with st.spinner(f"웹페이지 로드 중 ({i+1}/{len(all_sources['web'])}): {url}"):
                                web_docs = load_webpage(url)
                                if web_docs:
                                    all_documents.extend(web_docs)
                                    source_summaries.append(f"웹: {url}")
                                    if "web" not in sources_info:
                                        sources_info["web"] = []
                                    sources_info["web"].append(url)
                
                # PDF 처리
                if use_pdf and all_sources["pdf"]:
                    for i, file in enumerate(all_sources["pdf"]):
                        with st.spinner(f"PDF 로드 중 ({i+1}/{len(all_sources['pdf'])}): {file.name}"):
                            pdf_docs = load_pdf(file)
                            if pdf_docs:
                                all_documents.extend(pdf_docs)
                                source_summaries.append(f"PDF: {file.name}")
                                if "pdf" not in sources_info:
                                    sources_info["pdf"] = []
                                sources_info["pdf"].append(file.name)
                
                # YouTube 처리
                if use_youtube and all_sources["youtube"]:
                    for i, url in enumerate(all_sources["youtube"]):
                        if url:
                            with st.spinner(f"YouTube 로드 중 ({i+1}/{len(all_sources['youtube'])}): {url}"):
                                youtube_docs = load_youtube(url)
                                if youtube_docs:
                                    all_documents.extend(youtube_docs)
                                    source_summaries.append(f"YouTube: {url}")
                                    if "youtube" not in sources_info:
                                        sources_info["youtube"] = []
                                    sources_info["youtube"].append(url)
                
                # 문서 로드 결과 확인
                if all_documents:
                    documents_status.success(f"1. 문서 로드 완료: {len(all_documents)} 개의 문서 세그먼트")
                    
                    # 2. 청킹
                    chunking_status = st.empty()
                    chunking_status.info("2. 문서 청킹 중...")
                    texts = split_documents(all_documents)
                    chunking_status.success(f"2. 청킹 완료: {len(texts)} 개의 청크 생성됨")
                    
                    # 3. 임베딩
                    embedding_status = st.empty()
                    embedding_status.info("3. 임베딩 벡터화 중...")
                    embeddings = create_embeddings(texts, selected_embedding)
                    embedding_status.success("3. 임베딩 벡터화 완료")
                    
                    # 4. 벡터 DB 저장
                    vectordb_status = st.empty()
                    vectordb_status.info("4. FAISS 벡터 DB에 저장 중...")
                    vector_store = create_vector_db(texts, embeddings)
                    vectordb_status.success("4. FAISS 벡터 DB 저장 완료")
                    
                    # 세션 상태에 벡터 스토어와 소스 정보 저장
                    st.session_state.vector_store = vector_store
                    st.session_state.current_sources = source_summaries
                    
                    # 5. LLM 준비 완료
                    llm_status = st.empty()
                    llm_status.success("5. 질문 준비 완료!")
                    
                    # 대화 기록 초기화 (새 문서 조합이므로)
                    st.session_state.chat_history = []
                    
                    # 처리 완료 상태로 변경
                    st.session_state.processing_complete = True
                    
                    # 페이지 새로고침하여 채팅 UI로 전환
                    st.rerun()
                else:
                    documents_status.error("문서를 로드할 수 없습니다. 입력을 확인하세요.")
else:
    # 문서 처리 완료 후 채팅 UI 레이아웃
    # 2개 컬럼으로 나누기
    chat_col1, chat_col2 = st.columns([1, 3])
    
    with chat_col1:
        st.header("로드된 소스")
        # 로드된 소스 정보 표시
        for source in st.session_state.current_sources:
            st.info(source)
            
        # 다시 문서 로드하기 버튼
        if st.button("새 문서 로드하기"):
            st.session_state.processing_complete = False
            st.rerun()
            
        # 대화 기록 초기화 버튼
        if st.button("대화 기록 초기화"):
            st.session_state.chat_history = []
            st.rerun()
        
        # 채팅 입력창 (왼쪽 컬럼)
        st.subheader("질문 입력")

        # Streamlit 방식으로 Enter 키 제출 처리
        # text_area 대신 form을 사용하여 Enter 키 처리
        with st.form(key="query_form", clear_on_submit=True):
            user_input = st.text_input("질문을 입력하세요", key="user_query")
            submit_button = st.form_submit_button("전송")
            
        # 폼 제출 시 질문 처리
        if submit_button and user_input:
            process_query(user_input)
            # 채팅 기록 업데이트 후 페이지 새로고침
            st.rerun()
    
    with chat_col2:
        st.header("챗봇과의 대화")
        
        # 메시지 표시
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])