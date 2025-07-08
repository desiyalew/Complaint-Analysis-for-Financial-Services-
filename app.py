import streamlit as st
from src.rag_pipeline import RAGSystem

def main():
    st.set_page_config(page_title="CrediTrust Complaint Analyst", layout="wide")
    
    st.title("💬 CrediTrust Intelligent Complaint Analyzer")
    
    # Initialize RAG system
    @st.cache_resource
    def load_rag_system():
        return RAGSystem()
    
    rag_system = load_rag_system()
    
    # Sidebar with instructions
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This chatbot analyzes customer complaints for CrediTrust Financial using Retrieval-Augmented Generation.
    
    Ask questions about customer complaints such as:
    - Why are people unhappy with BNPL?
    - What are the common issues with credit cards?
    - Compare complaints between personal loans and BNPL services
    """)
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about customer complaints"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing complaints..."):
                answer, retrieved_docs = rag_system.run_rag(prompt)
                
                # Display answer
                st.markdown(answer)
                
                # Display sources
                with st.expander("Show sources"):
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Source {i+1}** (Product: {doc['metadata']['product']})")
                        st.markdown(f"Distance: {doc['distance']:.4f}")
                        st.markdown(f"{doc['text'][:200]}...")
                        st.markdown("---")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # Clear chat button
    if st.button("Clear Conversation"):
        st.session_state.messages = []

if __name__ == "__main__":
    main()