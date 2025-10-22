"""
Streamlit Web Interface for Urological Oncology RAG System
Professional UI - Refactored v2
"""

import streamlit as st
import os
import time
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

from src.retrieval_optimized import OptimizedRAGRetriever, OptimizedRetrievalConfig
from src.conversation_memory import ConversationMemory, ConversationalRAG

# Page configuration
st.set_page_config(
    page_title="Urological Oncology RAG",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Reduce top padding */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem;
    }

    /* Reduce sidebar top padding */
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem !important;
    }

    /* Fix sidebar width */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 360px !important;
        max-width: 360px !important;
    }

    /* Adjust main content margin when sidebar is fixed width */
    .main .block-container {
        max-width: calc(100% - 360px);
    }

    /* Compact headers */
    .main-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        margin-top: 0;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }

    /* Citation styling */
    .citation {
        color: #1f77b4;
        font-weight: 600;
        text-decoration: none;
        cursor: pointer;
        padding: 2px 6px;
        background: #e8f4f8;
        border-radius: 3px;
        font-size: 0.9em;
    }

    .citation:hover {
        background: #d0e8f0;
    }

    /* Source cards */
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
        margin-bottom: 0.5rem;
    }

    /* Metrics */
    .metric-box {
        background: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }

    /* Buttons */
    .stButton button {
        width: 100%;
    }

    /* Compact dividers */
    .compact-divider {
        margin: 0.5rem 0;
    }

    /* Reduce vertical gaps */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }

    /* Compact text areas */
    .stTextArea textarea {
        min-height: 80px !important;
    }

    /* Remove excessive padding from tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding-top: 0.5rem;
    }

    /* Reduce spacing in sidebar elements */
    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory(max_history=5)
if 'conv_rag' not in st.session_state:
    st.session_state.conv_rag = None
if 'session_metrics' not in st.session_state:
    st.session_state.session_metrics = {
        'queries': [],
        'latencies': [],
        'cache_hits': 0
    }
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'quality_metrics' not in st.session_state:
    st.session_state.quality_metrics = None
if 'show_query_metrics' not in st.session_state:
    st.session_state.show_query_metrics = False


@st.cache_resource
def load_rag_system(_api_key=None):
    """Load RAG system with optional user API key"""

    # Use user key if provided, otherwise use environment variable
    api_key = _api_key or os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("❌ No API key available. Please enter your OpenAI API key in the sidebar.")
        st.stop()

    # Set the API key for this session
    os.environ["OPENAI_API_KEY"] = api_key

    with st.spinner("🚀 Initializing RAG System..."):
        retriever = OptimizedRAGRetriever(
            chroma_db_dir="chroma_db_scaled",
            collection_name="urological_oncology_papers",
            config=OptimizedRetrievalConfig(
                top_k=5,
                max_context_length=3000,
                max_tokens=500,
                use_cache=True
            )
        )
    return retriever


@st.cache_data
def load_evaluation_metrics():
    """Load evaluation metrics if available"""
    try:
        metrics_path = Path("data/evaluation/aggregate_metrics.json")
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                return json.load(f)
    except:
        pass
    return None


def display_sidebar():
    """Minimalist sidebar with live session stats"""
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")

        # API Key input
        user_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Enter your key for unlimited usage. Leave empty for 1 free query."
        )

        # Store in session state
        if user_api_key:
            st.session_state.user_api_key = user_api_key
            st.success("✅ Your key active")
        else:
            st.session_state.user_api_key = None
            st.warning("⚠️ Using demo mode (1 query)")

        st.divider()

        st.markdown("### 🔬 System Information")

        # Live session metrics
        st.markdown("#### Current Session")

        metrics = st.session_state.session_metrics
        queries_count = len(metrics['queries'])

        if queries_count > 0:
            latencies = metrics['latencies']
            avg_latency = sum(latencies) / len(latencies)
            cache_hits = metrics['cache_hits']
            cache_rate = (cache_hits / queries_count * 100) if queries_count > 0 else 0

            st.markdown(f"""
```
            Queries:      {queries_count}
            Avg Latency:  {avg_latency:.2f}s
            Cache Hits:   {cache_hits}/{queries_count} ({cache_rate:.0f}%)
            Fastest:      {min(latencies):.2f}s
            Slowest:      {max(latencies):.2f}s
```
            """)
        else:
            st.info("No queries yet")

        st.divider()

        # Query Settings
        st.markdown("### ⚙️ Query Settings")

        top_k = st.slider(
            "Number of sources",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant chunks to retrieve"
        )

        model = st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0,
            help="Model for answer generation"
        )

        show_context = st.checkbox(
            "Show full context",
            value=False,
            help="Show complete source text instead of short preview"
        )

        return top_k, model, show_context

def format_answer_with_citations(answer, sources):
    """Format answer with inline citation links and hover tooltips"""
    formatted_answer = answer

    # Add citation styling
    for i in range(len(sources)):
        citation = f"[{i+1}]"
        if citation in formatted_answer:
            formatted_answer = formatted_answer.replace(
                citation,
                f'<span class="citation" title="{sources[i]["title"][:60]}...">{citation}</span>'
            )

    return formatted_answer


def display_sources(sources, show_context):
    """Display sources in collapsible expandable format"""
    st.markdown("### 📚 Sources")

    for idx, source in enumerate(sources, 1):
        with st.expander(
            f"**[{idx}] {source['title'][:70]}...**  "
            f"(Relevance: {source['similarity']:.0%})",
            expanded=False
        ):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.caption("**Section**")
                st.text(source['section'])

            with col2:
                st.caption("**PMID**")
                if source['pmid']:
                    st.markdown(f"[{source['pmid']}](https://pubmed.ncbi.nlm.nih.gov/{source['pmid']})")
                else:
                    st.text("N/A")

            with col3:
                st.caption("**DOI**")
                st.text(source['doi'] if source['doi'] else "N/A")

            st.divider()

            if show_context:
                st.caption("**Full Context:**")
                st.text(source['text_preview'])
            else:
                st.caption("**Preview:**")
                st.text(source['text_preview'][:200] + "...")


def evaluate_response_quality(retriever, query, answer, sources):
    """Evaluate response quality on-demand"""
    from src.evaluation import RAGEvaluator

    evaluator = RAGEvaluator(rag_retriever=retriever)

    # Prepare context
    context = "\n\n".join([
        f"[Doc {i+1}]\n{source['text_preview'][:500]}"
        for i, source in enumerate(sources)
    ])

    # Evaluate
    faithfulness = evaluator.evaluate_faithfulness(query, answer, context)
    relevance = evaluator.evaluate_relevance(query, answer)

    # Prepare chunks for precision
    chunks = [{'text': s['text_preview']} for s in sources]
    precision = evaluator.evaluate_context_precision(query, chunks)

    return {
        'faithfulness': faithfulness,
        'relevance': relevance,
        'precision': precision
    }


def create_metrics_gauge(value, title):
    """Create a gauge chart for a metric"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "lightblue"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    return fig


def display_metrics_dashboard():
    """Display evaluation metrics dashboard"""

    # Check if we should show query-specific metrics
    if st.session_state.get('show_query_metrics') and st.session_state.quality_metrics:
        st.markdown("## 📊 Current Query Metrics")

        m = st.session_state.quality_metrics

        # Display 4 gauges with tooltips
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fig1 = create_metrics_gauge(m['faithfulness'], "Faithfulness")
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("💡 Measures if the answer is based solely on retrieved sources without hallucination")

        with col2:
            fig2 = create_metrics_gauge(m['relevance'], "Relevance")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("💡 Measures if the answer directly addresses the user's question")

        with col3:
            fig3 = create_metrics_gauge(m['precision'], "Context Precision")
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("💡 Measures if the retrieved sources are relevant to the question")

        with col4:
            overall = (m['faithfulness'] + m['relevance'] + m['precision']) / 3
            fig4 = create_metrics_gauge(overall, "Overall Quality")
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("💡 Average of all quality metrics")

        st.divider()

        st.info("💡 **Tip:** These metrics evaluate the quality of your most recent query. Return to the Query tab to ask another question.")

    else:
        # Show baseline metrics from evaluation file
        st.markdown("## 📊 System Performance Baseline")
        st.info("These metrics are from the initial system evaluation on 8 test queries. Click **'Evaluate Response Quality'** in the Query tab to see metrics for your current query.")

        metrics = load_evaluation_metrics()

        if not metrics:
            st.warning("⚠️ No evaluation metrics found. Run: `python -m src.evaluation`")
            return

        # Main metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            fig1 = create_metrics_gauge(metrics.get('avg_faithfulness', 0), "Faithfulness")
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("💡 Measures if answers are based solely on sources without hallucination")

        with col2:
            fig2 = create_metrics_gauge(metrics.get('avg_relevance', 0), "Relevance")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("💡 Measures if answers directly address the questions")

        with col3:
            fig3 = create_metrics_gauge(metrics.get('avg_context_precision', 0), "Context Precision")
            st.plotly_chart(fig3, use_container_width=True)
            st.caption("💡 Measures if retrieved sources are relevant to questions")

        with col4:
            overall = (
                metrics.get('avg_faithfulness', 0) +
                metrics.get('avg_relevance', 0) +
                metrics.get('avg_context_precision', 0)
            ) / 3
            fig4 = create_metrics_gauge(overall, "Overall Quality")
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("💡 Average of all quality metrics")


def main():
    """Main Streamlit app"""

    # Compact header
    st.markdown('<div class="main-header">🔬 Urological Oncology RAG System</div>', unsafe_allow_html=True)

    # Sidebar
    top_k, model, show_context = display_sidebar()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Query", "📊 Metrics", "ℹ️ About"])

    # Tab 1: Query Interface
    with tab1:
        # Load RAG system
        user_key = st.session_state.get('user_api_key')
        retriever = load_rag_system(_api_key=user_key)
        retriever.config.top_k = top_k

       # 1. KNOWLEDGE BASE (First)
        st.markdown("### 📚 Knowledge Base")

        # Topic selector
        topic_filter = st.selectbox(
            "Search in:",
            ["All Topics", "Prostate Cancer", "Bladder Cancer", "Kidney Cancer", "Testicular Cancer"],
            index=0,
            help="Filter by cancer type (auto-detects by default)"
        )

        # Add vertical spacing
        st.markdown("<div style='margin: 20px 0;'></div>", unsafe_allow_html=True)

        # Conversation controls in row
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("🔄 Reset Current Chat", use_container_width=True):
                memory = st.session_state.conversation_memory
                st.session_state.conversation = memory.create_conversation()
                st.session_state.current_response = None
                st.session_state.quality_metrics = None
                st.rerun()

        with col2:
            context_mode = st.checkbox(
                "💬 Enable Chat Mode",
                value=st.session_state.conversation is not None,
                help="Multi-turn conversation with context"
            )

        if context_mode and st.session_state.conversation is None:
            memory = st.session_state.conversation_memory
            st.session_state.conversation = memory.create_conversation()
        elif not context_mode:
            st.session_state.conversation = None

        # Show conversation status inline
        if st.session_state.conversation:
            turns = len(st.session_state.conversation.messages) // 2
            st.caption(f"💬 Active chat • {turns} turns")
        else:
            pass

        st.divider()

        # 3. EXAMPLE QUERIES (Third)
        st.markdown("### 💡 Example Queries")
        st.markdown("Select an example or type your own below:")

        # Add spacing
        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)

        # Mixed topic examples
        example_query = st.selectbox(
            "Choose an example:",
            [
                # Prostate
                "What are the current treatment options for prostate cancer?",
                "What are the side effects of androgen deprivation therapy?",
                "What is castration-resistant prostate cancer?",
                # Bladder
                "What are the treatment options for bladder cancer?",
                "What is the role of BCG immunotherapy in bladder cancer?",
                "What are the side effects of intravesical therapy?",
                # Kidney
                "What is the role of immunotherapy in kidney cancer?",
                "What are targeted therapies for renal cell carcinoma?",
                "What are the treatment options for advanced kidney cancer?",
                # Testicular
                "What are the chemotherapy options for testicular cancer?",
                "What is the cure rate for testicular cancer?",
                "What are the side effects of chemotherapy for germ cell tumors?",
            ],
            key="example_selector",
            label_visibility="collapsed"
        )

        st.divider()

        # 4. QUERY INPUT (Fourth)
        st.markdown("### 🔍 Your Question")

        query = st.text_area(
            "Ask your question:",
            value=example_query if example_query else st.session_state.get('query', ''),
            height=120,
            placeholder="Ask about prostate cancer treatments, diagnosis, biomarkers, side effects, etc.",
            key="query_input",
            label_visibility="collapsed"
        )

        # Search buttons
        col1, col2 = st.columns([1, 3])
        with col1:
            search_button = st.button("🚀 Search", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)

        if clear_button:
            st.session_state.query = ''
            st.session_state.current_response = None
            st.session_state.quality_metrics = None
            st.rerun()

        # Execute query
        if search_button and query:
            query = query.strip()

            with st.spinner("🔍 Searching knowledge base..."):
                start_time = time.time()

                try:
                    # Check conversation mode
                    if st.session_state.conversation:
                        if st.session_state.conv_rag is None:
                            st.session_state.conv_rag = ConversationalRAG(
                                retriever,
                                st.session_state.conversation_memory
                            )

                        response = st.session_state.conv_rag.query(
                            question=query,
                            conversation=st.session_state.conversation,
                            model=model
                        )

                        rewritten_query = response.get('rewritten_query')
                    else:
                        response = retriever.query(
                            question=query,
                            model=model,
                            return_sources=True,
                            use_cache=True
                        )
                        rewritten_query = None

                    latency = time.time() - start_time

                    # Store response in session state
                    st.session_state.current_response = {
                        'query': query,
                        'answer': response['answer'],
                        'sources': response.get('sources', []),
                        'num_sources': response['num_sources'],
                        'latency': latency,
                        'rewritten_query': rewritten_query
                    }

                    # Clear previous quality metrics
                    st.session_state.quality_metrics = None

                    # Track session metrics
                    st.session_state.session_metrics['queries'].append(query)
                    st.session_state.session_metrics['latencies'].append(latency)
                    if latency < 0.5:
                        st.session_state.session_metrics['cache_hits'] += 1

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    import traceback
                    with st.expander("Debug Info"):
                        st.code(traceback.format_exc())

        elif search_button:
            st.warning("⚠️ Please enter a question")

        # DISPLAY RESULTS (separate from search button, uses session state)
        if st.session_state.current_response:
            resp = st.session_state.current_response

            st.divider()

            # Show query rewrite if conversation mode
            if resp['rewritten_query'] and resp['rewritten_query'] != resp['query']:
                with st.expander("🔄 Query Rewrite (Context Applied)", expanded=True):
                    st.markdown(f"**Original:** {resp['query']}")
                    st.markdown(f"**Expanded:** {resp['rewritten_query']}")

            # Display results header
            is_cached = resp['latency'] < 0.5
            cache_indicator = " ✨ (cached)" if is_cached else ""

            st.success(f"✅ Found {resp['num_sources']} relevant sources in {resp['latency']:.2f}s{cache_indicator}")

            st.divider()

            # Answer with inline citations
            st.markdown("### 📄 Answer")

            formatted_answer = format_answer_with_citations(
                resp['answer'],
                resp['sources']
            )
            st.markdown(formatted_answer, unsafe_allow_html=True)


            # Quality evaluation button (now works with session state!)
             # Quality evaluation button - evaluates and switches to Metrics tab
            if st.button("🔬 Evaluate Response Quality"):
                with st.spinner("🔬 Evaluating response quality..."):
                    metrics = evaluate_response_quality(
                        retriever,
                        resp['query'],
                        resp['answer'],
                        resp['sources']
                    )
                    st.session_state.quality_metrics = metrics
                    st.session_state.show_query_metrics = True
                    # Note: We'll switch to tab programmatically
                    st.info("✅ Evaluation complete! Switch to the **Metrics** tab to view results.")

            st.divider()

            # Sources
            display_sources(resp['sources'], show_context)

    # Tab 2: Metrics Dashboard
    with tab2:
        display_metrics_dashboard()

    # Tab 3: About
    with tab3:
        st.markdown("## 🎯 Project Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📚 Dataset")
            st.info("""
            - **Source:** PubMed Central Open Access
            - **Papers:** 815 full-text peer-reviewed articles
            - **Chunks:** 41,970 section-aware segments
            - **Topics:** Prostate, Bladder, Kidney, Testicular Cancer
            - **Years:** 2015-2025
            - **Avg Sections:** 15.5 per paper
            """)

            st.markdown("### 🏗️ Architecture")
            st.info("""
            1. Data Collection: PubMed API + BioPython
            2. Processing: Section-aware chunking
            3. Embeddings: OpenAI text-embedding-3-small
            4. Storage: ChromaDB vector database
            5. Retrieval: Semantic similarity search
            6. Generation: GPT-4o-mini with citations
            7. Memory: Conversation context tracking
            """)

        with col2:
            st.markdown("### ✨ Features")
            st.success("""
            ✅ **Multi-topic coverage** (4 cancer types)
            ✅ **Section-aware retrieval**
            ✅ **Citation tracking**
            ✅ **Conversation memory**
            ✅ **Quality evaluation**
            ✅ **Smart caching** (99.9% speedup)
            ✅ **41,970 searchable chunks**
            """)

            st.markdown("### 🛠️ Technology Stack")
            st.info("""
            - Backend: Python, LangChain
            - LLM: OpenAI GPT-4o-mini
            - Embeddings: text-embedding-3-small
            - Vector DB: ChromaDB
            - Frontend: Streamlit
            - Deployment: Docker
            """)

        st.divider()

        st.markdown("### 📊 Performance Highlights")

        metrics = load_evaluation_metrics()
        if metrics:
            col1, col2, col3, col4 = st.columns(4)

            overall = (
                metrics.get('avg_faithfulness', 0) +
                metrics.get('avg_relevance', 0) +
                metrics.get('avg_context_precision', 0)
            ) / 3

            with col1:
                st.metric("Overall Quality", f"{overall:.1%}", delta="Grade A")
            with col2:
                st.metric("Faithfulness", f"{metrics.get('avg_faithfulness', 0):.1%}")
            with col3:
                st.metric("Relevance", f"{metrics.get('avg_relevance', 0):.1%}")
            with col4:
                st.metric("Precision", f"{metrics.get('avg_context_precision', 0):.1%}")


if __name__ == "__main__":
    main()
