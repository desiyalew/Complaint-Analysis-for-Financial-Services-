# src/chunking_embedding.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import pandas as pd
import os

class ComplaintVectorizer:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize the vectorizer with embedding model and text splitter"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="../vector_store/")
        
        # Define embedding function for ChromaDB
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        
        # Create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="complaint_vectors",
            embedding_function=self.sentence_transformer_ef
        )
    
    def chunk_text(self, text):
        """Split a single text into chunks"""
        return self.text_splitter.split_text(text)
    
    def process_dataframe(self, df):
        """Process the entire dataframe of complaints"""
        ids = []
        documents = []
        metadatas = []
        
        for idx, row in df.iterrows():
            chunks = self.chunk_text(row['cleaned_narrative'])
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{row.name}_{i}"
                
                ids.append(doc_id)
                documents.append(chunk)
                metadatas.append({
                    'original_id': row.name,
                    'product': row['Product'],
                    'company': row['Company'],
                    'issue': row['Issue']
                })
        
        return ids, documents, metadatas
    
    def store_embeddings(self, ids, documents, metadatas):
        """Store embeddings in ChromaDB in smaller batches to avoid size limits"""
        batch_size = 1000  # Adjust based on system capability and ChromaDB limits
        total_items = len(ids)
        
        print(f"Storing {total_items} document chunks in vector database in batches of {batch_size}")
        
        for i in range(0, total_items, batch_size):
            batch_end = min(i + batch_size, total_items)
            print(f"Adding batch {i} to {batch_end}...")
            
            self.collection.add(
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
        
        print(f"Finished storing {self.collection.count()} items")
    
def main():
    # Load preprocessed data
    df = pd.read_csv('../data/filtered_complaints.csv')
    
    # Initialize vectorizer
    vectorizer = ComplaintVectorizer()
    
    # Process dataframe into chunks
    ids, documents, metadatas = vectorizer.process_dataframe(df)
    
    # Store embeddings in vector database
    vectorizer.store_embeddings(ids, documents, metadatas)
    
    # Save chunking strategy and model info
    with open('../reports/vectorization_report.txt', 'w') as f:
        f.write("Text Chunking and Embedding Report\n")
        f.write("==================================\n\n")
        
        f.write("Chunking Strategy:\n")
        f.write("- Used LangChain's RecursiveCharacterTextSplitter\n")
        f.write("- Chunk size: 512 tokens\n")
        f.write("- Overlap: 50 tokens\n")
        f.write("- Splitting on: ['\\n\\n', '\\n', '.', ' ', '']\n")
        f.write("- Length function: character-based\n\n")
        
        f.write("Embedding Model:\n")
        f.write("- Model Name: sentence-transformers/all-MiniLM-L6-v2\n")
        f.write("- Description: A lightweight, fast, and efficient sentence transformer model suitable for semantic search.\n")
        f.write("- Dimension: 384\n")
        f.write("- Benefits: Good balance between performance and accuracy for semantic similarity tasks.\n")
    
    print("Vector database created and stored successfully")

if __name__ == "__main__":
    main()