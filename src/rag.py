from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# ==================================================
# 1. Load Embedding Model
# ==================================================

embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_name
)


# ==================================================
# 2. Load Existing FAISS Vector Store
# ==================================================

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)


# ==================================================
# 3. Load LLM
# ==================================================

llm_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    llm_model_name
)

model = AutoModelForCausalLM.from_pretrained(
    llm_model_name
)


# ==================================================
# 4. RAG Question Answering Function
# ==================================================

def ask_question(question):

    # --------------------------------------------------
    # Retrieve relevant documents
    # --------------------------------------------------

    results = vectorstore.similarity_search_with_score(
        question,
        k=3
    )


    # --------------------------------------------------
    # Create context from retrieved documents
    # --------------------------------------------------

    context = "\n\n".join(
        doc.page_content
        for doc, score in results
    )


    # --------------------------------------------------
    # Create RAG Prompt
    # --------------------------------------------------

    prompt = f"""You are a helpful customer support assistant.

Answer the question using only the information provided in the context.

If the answer is not available in the context, say:
"I don't have enough information in the provided documents."

Context:
{context}

Question:
{question}

Answer:"""


    # --------------------------------------------------
    # Create Chat Input
    # --------------------------------------------------

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]


    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True
    )


    # --------------------------------------------------
    # Generate Response using SmolLM2
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )


    # --------------------------------------------------
    # Extract Only Newly Generated Tokens
    # --------------------------------------------------

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]


    # --------------------------------------------------
    # Decode Generated Tokens
    # --------------------------------------------------

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()


    # --------------------------------------------------
    # Remove Unwanted "assistant" Prefix
    # --------------------------------------------------

    if answer.startswith("assistant"):
        answer = answer[len("assistant"):].strip()


    return answer


# ==================================================
# 5. Test RAG Pipeline
# ==================================================

if __name__ == "__main__":

    question = input("\nAsk a question: ")

    answer = ask_question(question)

    print("\nFinal Answer:")
    print("=" * 80)
    print(answer)