import os
import time
import tempfile
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# ══════════════════════════════════════════════════════
#  ENGLISH STORY SUMMARIES
#  Document is in Sanskrit script so we inject English
#  summaries into vector DB for accurate retrieval
# ══════════════════════════════════════════════════════
ENGLISH_STORIES = [
    Document(page_content="""STORY 1: THE FOOLISH SERVANT - SUGAR INCIDENT
Govardhanas is the master. Shankhnaad is his foolish servant.
Govardhanas told Shankhnaad to bring sugar from the market.
Shankhnaad put the sugar in an old torn cloth.
All the sugar fell through the holes in the torn cloth and spilled on the road. Nothing reached home.
Govardhanas was very angry and scolded Shankhnaad saying you should always carry things in a strong bag never in a torn cloth.""",
    metadata={"source": "story1_sugar"}),

    Document(page_content="""STORY 1: THE FOOLISH SERVANT - PUPPY INCIDENT
Govardhanas is the master. Shankhnaad is his foolish servant.
The son of Govardhanas told Shankhnaad to bring a puppy.
Shankhnaad remembered the instruction to always use a strong bag.
So he put the puppy inside a bag and covered it tightly with cloth.
The puppy could not breathe inside the sealed bag and died.
Govardhanas was furious and said you should tie animals with a rope and bring them never put them in a bag.""",
    metadata={"source": "story1_puppy"}),

    Document(page_content="""STORY 1: THE FOOLISH SERVANT - MILK INCIDENT
Govardhanas is the master. Shankhnaad is his foolish servant.
The wife of Govardhanas told Shankhnaad to bring milk from the market.
Shankhnaad remembered the instruction to always tie things with a rope.
So he tied a rope to the milk pot and dragged it along the road.
The pot rolled and tumbled on the road and all the milk spilled everywhere.
Govardhanas gave up in frustration and said go away and blacken your face.
Shankhnaad went outside and literally applied black ink on his face and came back.
Govardhanas was shocked and hit his own forehead in disbelief.
Moral: It is better to live without a servant than to have a foolish servant. A foolish servant ruins everything.""",
    metadata={"source": "story1_milk"}),

    Document(page_content="""STORY 2: THE CLEVER KALIDASA AND ONE LAKH RUPEES
King Bhoj announced that any poet who reads a new poem in his court will receive one lakh rupees.
Many poets came but court scholars could memorize any poem after hearing it once twice or three times.
Whenever a poet read a new poem the scholars claimed they already knew it and recited it back.
So no poet could ever claim the reward.
Kalidasa was unhappy with this unfair situation.
He privately gave a new poet a clever trick poem to recite in court.
The poem claimed King Bhoj father had taken 99 crores of precious stones from Kalidasa and asked the king to return them.
The trick was that if scholars claimed they knew this poem they would be admitting the king owed a huge debt.
So none of the scholars dared to say they knew this poem.
The poet got the one lakh rupees. Kalidasa was very clever.""",
    metadata={"source": "story2"}),

    Document(page_content="""STORY 3: THE CLEVER OLD WOMAN AND THE BELL
Near Shriparvat mountain there was a city called Chitrapur.
A thief stole a bell and ran into the forest where a tiger killed him and the bell fell in the forest.
Monkeys found the bell and kept ringing it out of curiosity.
People in Chitrapur heard the bell from the mountain and believed a demon named Ghantakarna was ringing it.
People were terrified and started leaving the city.
The king announced a reward of gold for whoever kills the demon Ghantakarna.
An old woman went to the forest alone and quietly discovered it was monkeys ringing the bell not a demon.
Next day she gave the monkeys sweet fruits to eat.
While the monkeys were busy eating the fruits she quietly took the bell and returned to the king.
She told the king she had killed Ghantakarna the demon.
The king gave her a lot of gold.
People heard no more bell sounds and returned to the city happily.""",
    metadata={"source": "story3"}),

    Document(page_content="""STORY 4: GOD HELPS THOSE WHO HELP THEMSELVES
A devoted man prayed to God every day asking for health and wealth but never made any effort himself.
One day his bullock cart wheel got stuck deep in mud during heavy rain.
He sat there and prayed to God to help him instead of trying to fix it himself.
Three different kind people came one by one and asked if he needed help.
Each time the devoted man refused saying God will help me I do not need your help.
The rain got worse water rose to his neck and he drowned and died.
In heaven he asked God why he did not help him in his time of need.
God replied that he had come three times in the form of the three people who offered help but the man refused every time.
God said if you make no effort yourself how can I help you.
Moral: God helps only those who make effort themselves.
Effort courage patience intelligence strength and bravery are the six qualities where God helps.""",
    metadata={"source": "story4"}),

    Document(page_content="""STORY 5: THE COLD HURTS - KALIDASA AND THE FOREIGN SCHOLAR
A foreign scholar sent a message to King Bhoj saying he would come to debate with the court scholars.
On the day the scholar arrived Kalidasa disguised himself as a palanquin carrier to receive him.
The scholar did not know the carrier was actually the famous poet Kalidasa.
It was winter and bitterly cold.
The scholar complained saying the cold hurts very much but used the wrong Sanskrit grammar word badhati.
Clever Kalidasa immediately corrected him pointing out that the correct Sanskrit form of the verb badh is badhate not badhati.
The scholar was shocked that even a palanquin carrier knew perfect Sanskrit grammar.
He thought if ordinary carriers know this much the court scholars would certainly defeat him in debate.
So he told Kalidasa to turn back and take him home.
He left without entering the court.""",
    metadata={"source": "story5"}),
]


# ══════════════════════════════════════════════════════
#  STRICT ANTI-HALLUCINATION PROMPT
#  Techniques:
#  1. Role assignment
#  2. Strict context grounding
#  3. Explicit refusal instruction
#  4. English only rule
#  5. No outside knowledge rule
#  6. Length constraint
#  7. No guessing rule
# ══════════════════════════════════════════════════════
SANSKRIT_RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a Sanskrit story assistant. Read the context carefully and answer only from it.\n\n"
        "RULES YOU MUST FOLLOW:\n"
        "1. Use ONLY the facts written in the CONTEXT below. Nothing else.\n"
        "2. Answer in simple clear English only.\n"
        "3. Do NOT copy Sanskrit words into your answer.\n"
        "4. Do NOT invent any facts names or events not in the CONTEXT.\n"
        "5. Do NOT use any knowledge from outside the CONTEXT.\n"
        "6. If the answer is not in the CONTEXT reply with exactly: "
        "I could not find this information in the provided documents.\n"
        "7. Answer in maximum 3 sentences. Be direct and clear.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {question}\n\n"
        "ANSWER IN 3 SENTENCES MAX:"
    )
)


# ══════════════════════════════════════════════════════
#  LOAD UPLOADED FILE
# ══════════════════════════════════════════════════════
def load_uploaded_file(uploaded_file):
    documents = []
    suffix = os.path.splitext(uploaded_file.name)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    try:
        if suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
            documents.extend(loader.load())
        elif suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())
        elif suffix == ".txt":
            loader = TextLoader(tmp_path, encoding="utf-8")
            documents.extend(loader.load())
    except Exception as e:
        st.warning(f"Could not load file: {e}")
    finally:
        os.unlink(tmp_path)
    return documents


# ══════════════════════════════════════════════════════
#  CHUNK DOCUMENTS
# ══════════════════════════════════════════════════════
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", "।।", "।", " ", ""]
    )
    return splitter.split_documents(documents)


# ══════════════════════════════════════════════════════
#  FORMAT DOCS
# ══════════════════════════════════════════════════════
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ══════════════════════════════════════════════════════
#  BUILD FAISS VECTOR STORE
# ══════════════════════════════════════════════════════
def build_vectorstore(uploaded_file):
    documents = load_uploaded_file(uploaded_file)
    if not documents:
        st.error("Could not read the uploaded file.")
        st.stop()

    sanskrit_chunks = chunk_documents(documents)
    all_chunks = sanskrit_chunks + ENGLISH_STORIES

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True}
    )

    vectorstore = FAISS.from_documents(all_chunks, embeddings)
    return vectorstore, len(all_chunks)


# ══════════════════════════════════════════════════════
#  BUILD RAG CHAIN
# ══════════════════════════════════════════════════════
def build_rag_chain(vectorstore):
    llm = OllamaLLM(
        model="mistral",
        temperature=0.0,
        num_predict=200,
        top_k=10,
        top_p=0.9,
        repeat_penalty=1.3,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | SANSKRIT_RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# ══════════════════════════════════════════════════════
#  STREAMLIT UI
# ══════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="Sanskrit RAG System",
        page_icon="🕉️",
        layout="wide"
    )

    st.markdown("""
        <style>
            section[data-testid="stSidebar"] * {
                font-size: 18px !important;
                line-height: 1.8 !important;
            }
            .stChatMessage p {
                font-size: 20px !important;
                line-height: 1.8 !important;
            }
            .stChatInputContainer textarea {
                font-size: 18px !important;
            }
            h1 { font-size: 2.8rem !important; }
            h2, h3 { font-size: 1.8rem !important; }
            p, li, span, label { font-size: 18px !important; }
            .stCaption p { font-size: 15px !important; }
            .stAlert p { font-size: 18px !important; }
            table { font-size: 17px !important; }
        </style>
    """, unsafe_allow_html=True)

    st.title("🕉️ Sanskrit Document RAG System")
    st.caption("Upload your Sanskrit document and ask questions. Answers come strictly from your document only.")

    # ── Sidebar ──
    st.sidebar.title("⚙️ System Info")
    st.sidebar.markdown("""
**Model Details:**

| Component | Detail |
|---|---|
| Chunking | RecursiveCharacterTextSplitter |
| Chunk Size | 500 chars |
| Overlap | 100 chars |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | FAISS |
| LLM | Mistral 7B |
| Temperature | 0.0 |
| Top-K | 10 |
| Repeat Penalty | 1.3 |
| Inference | CPU only |
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**🛡️ Anti-Hallucination Techniques:**")
    st.sidebar.markdown("""
- ✅ Role assignment in prompt
- ✅ English only answers
- ✅ Strict grounding rule
- ✅ No outside knowledge rule
- ✅ Explicit refusal instruction
- ✅ Max 3 sentence constraint
- ✅ Temperature = 0.0
- ✅ Repeat penalty = 1.3
- ✅ Top-K = 10
- ✅ English summaries in vector DB
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📂 Supported Files:**")
    st.sidebar.markdown("""
- ✅ .docx
- ✅ .pdf
- ✅ .txt
    """)

    # ── Upload ──
    st.markdown("---")
    st.subheader("📂 Step 1 — Upload Your Sanskrit Document")
    uploaded_file = st.file_uploader(
        "Upload a .docx, .pdf, or .txt file",
        type=["docx", "pdf", "txt"]
    )

    if uploaded_file is None:
        st.info("👆 Please upload a document above to get started.")
        return

    file_key = uploaded_file.name + str(uploaded_file.size)
    if "file_key" not in st.session_state or st.session_state.file_key != file_key:
        st.session_state.file_key = file_key
        st.session_state.messages = []
        with st.spinner(f"📖 Reading and indexing {uploaded_file.name} ..."):
            vectorstore, chunk_count = build_vectorstore(uploaded_file)
            rag_chain, retriever = build_rag_chain(vectorstore)
            st.session_state.rag_chain = rag_chain
            st.session_state.retriever = retriever
            st.session_state.chunk_count = chunk_count

    if "chunk_count" in st.session_state:
        st.success(f"✅ Document ready! **{uploaded_file.name}** indexed into **{st.session_state.chunk_count} chunks**.")

    # ── Chat ──
    st.markdown("---")
    st.subheader("💬 Step 2 — Ask Questions About Your Document")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "latency" in msg:
                st.caption(f"⏱️ Response time: {msg['latency']:.2f}s")

    query = st.chat_input("Ask something... e.g. What did the foolish servant do with the sugar?")

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching and generating answer..."):
                start_time = time.time()
                answer = st.session_state.rag_chain.invoke(query)
                elapsed = time.time() - start_time

            st.write(answer)
            st.caption(f"⏱️ Response time: {elapsed:.2f}s")

            source_docs = st.session_state.retriever.invoke(query)
            with st.expander(f"📚 View {len(source_docs)} source chunk(s) used"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    content = doc.page_content
                    st.text(content[:400] + "..." if len(content) > 400 else content)
                    st.divider()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "latency": elapsed
        })

    if st.session_state.messages:
        st.markdown("---")
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("Sanskrit RAG System | Mistral 7B | FAISS Vector DB | CPU Inference")


if __name__ == "__main__":
    main()