import os
import sys
import lancedb
from openai import OpenAI


def generate_query_embedding(query_text, api_key):
    """Generate embedding for the query text using NVIDIA's embedding model."""
    client = OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1"
    )
    
    response = client.embeddings.create(
        input=[query_text],
        model="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
        encoding_format="float",
        extra_body={"modality": ["text"], "input_type": "query", "truncate": "NONE"}
    )
    
    return response.data[0].embedding


def search_lancedb(query_embedding, db_path="lancedb", limit=5):
    """Search LanceDB for relevant documents using vector similarity."""
    db = lancedb.connect(db_path)
    table = db.open_table("document_embeddings")
    
    results = table.search(query_embedding).limit(limit).to_list()
    
    return results


def generate_answer(query, context_docs, api_key):
    """Generate answer using NVIDIA's Nemotron model with retrieved context."""
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        source = doc.get("source_id", "Unknown")
        page = doc.get("chunk_sequence", "?")
        text = doc.get("text", "")
        context_parts.append(f"[Source: {source}, Page {page}]\n{text}")
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Create prompt with context
    prompt = f"""Based on the following document excerpts, please answer the question.

Context:
{context}

Question: {query}

Please provide a detailed answer based on the context above. If the context doesn't contain enough information to answer the question, please say so."""
    
    # Call Nemotron model
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    
    completion = client.chat.completions.create(
        model="nvidia/nvidia-nemotron-nano-9b-v2",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        frequency_penalty=0,
        presence_penalty=0,
        stream=False,
        extra_body={
            "min_thinking_tokens": 1024,
            "max_thinking_tokens": 2048
        }
    )
    
    reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
    answer = completion.choices[0].message.content
    
    return answer, reasoning, context_docs


def main():
    # Get API key
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: NVIDIA_API_KEY environment variable not set")
        sys.exit(1)
    
    # Get query from command line or prompt
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your question: ")
    
    if not query.strip():
        print("Error: No query provided")
        sys.exit(1)
    
    print(f"\nQuery: {query}\n")
    print("=" * 80)
    
    # Step 1: Generate embedding for query
    print("\n[1/3] Generating query embedding...")
    query_embedding = generate_query_embedding(query, api_key)
    
    # Step 2: Search LanceDB for relevant documents
    print("[2/3] Searching for relevant documents...")
    results = search_lancedb(query_embedding)
    
    print(f"\nFound {len(results)} relevant documents:")
    for i, doc in enumerate(results, 1):
        source = doc.get("source_id", "Unknown")
        page = doc.get("chunk_sequence", "?")
        distance = doc.get("_distance", 0)
        print(f"  {i}. {source} (Page {page}) - Distance: {distance:.4f}")
    
    # Step 3: Generate answer with context
    print("\n[3/3] Generating answer...")
    answer, reasoning, context_docs = generate_answer(query, results, api_key)
    
    print("\n" + "=" * 80)
    
    if reasoning:
        print("\n🤔 REASONING:")
        print("-" * 80)
        print(reasoning)
        print("\n" + "=" * 80)
    
    print("\n💡 ANSWER:")
    print("-" * 80)
    print(answer)
    print("\n" + "=" * 80)
    
    # Show sources
    print("\n📚 SOURCES:")
    print("-" * 80)
    for i, doc in enumerate(context_docs, 1):
        source = doc.get("source_id", "Unknown")
        page = doc.get("chunk_sequence", "?")
        print(f"  {i}. {source} - Page {page}")
    print()


if __name__ == "__main__":
    main()
