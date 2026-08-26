import os
import pandas as pd
from ragas import evaluate, EvaluationDataset
# Import the classes instead of the functions
from ragas.metrics import Faithfulness, AnswerRelevancy 
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

# --- CRITICAL: PREVENT OPENAI CRASH ---
os.environ["OPENAI_API_KEY"] = "LOCAL_EVALUATION_MODE"

# 1. Initialize LOCAL Wrappers
eval_llm = LangchainLLMWrapper(OllamaLLM(model="llama3", temperature=0))
local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ragas_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

# 2. Initialize the Metric OBJECTS
# We pass our local LLM and Embeddings directly into the metrics
faithfulness = Faithfulness(llm=eval_llm)
answer_relevancy = AnswerRelevancy(llm=eval_llm, embeddings=ragas_embeddings)

# 3. Define the Test Dataset (Same as before)
test_data = [
    {
        "user_input": "இயற்கை விவசாயம் பஞ்சகவ்யம் தயாரிப்பது எப்படி?",
        "retrieved_contexts": ["To prepare 20 liters of Panchagavya, mix 5 kg cow dung with 1 kg ghee. After 3 days add 3L urine, 2L milk, 2L curd..."],
        "response": "To prepare 20 liters of Panchagavya, you need 5kg cow dung, 1kg ghee, 3L urine, 2L milk, and 2L curd. Mix dung and ghee for 3 days first.",
        "reference": "The standard 20L recipe requires 5kg dung, 1kg ghee, 3L urine, 2L milk, 2L curd, 3L coconut water, and 12 bananas."
    },
    {
        "user_input": "What is intensive farming?",
        "retrieved_contexts": ["Intensive farming involves high input of capital and labor relative to land area to maximize crop yield."],
        "response": "Intensive farming is a system of cultivation using large amounts of labor and capital relative to land area to increase yields.",
        "reference": "Intensive farming is an agricultural intensification and mechanization system that aims to maximize yields from available land."
    }
]

def run_eval():
    print("⚖️ Starting RAGAS Evaluation (Fully Local - v0.2 Protocol)...")
    ds = EvaluationDataset.from_list(test_data)
    
    try:
        # Pass the initialized metric objects
        results = evaluate(
            dataset=ds,
            metrics=[faithfulness, answer_relevancy]
        )
        
        print("\n✅ Evaluation Successful!")
        print(results)
        
        df = results.to_pandas()
        df.to_csv("research_metrics.csv", index=False)
        print("📊 Results saved to 'research_metrics.csv'")

    except Exception as e:
        print(f"❌ Evaluation Failed: {e}")

if __name__ == "__main__":
    run_eval()