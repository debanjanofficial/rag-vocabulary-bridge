pip install langchain langchain-community pypdf pandas
import os
import json
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = ChatOllama(model="llama3", temperature=0.2, format="json")

prompt = ChatPromptTemplate.from_template("""
You are an expert terminology manager. Read the following technical text from a product manual.
Based on the text, generate:
1. A formal, highly technical query (Reference Query) someone might ask about this text.
2. A colloquial, informal, or messy query (User Query) an everyday customer might ask.
3. A list of terminology mappings linking the informal terms to the formal technical terms.

Text: {text}

Output valid JSON strictly in this format:
{{
    "formal_query": "...",
    "colloquial_query": "...",
    "mappings": [{{"informal": "...", "formal": "..."}}]
}}
""")

chain = prompt | llm | JsonOutputParser()

# --- YOUR EXACT PATHS ---
RAW_DIR = r"C:\Users\MikeB\Desktop\Academy\FAU\Semester 4\ABA\Dataset\RAUVISIO_House_cleaned"
CSV_OUT = r"C:\Users\MikeB\Desktop\Academy\FAU\Semester 4\ABA\Dataset\Evaluation_Dataset_with_Sources.csv"
JSON_OUT = r"C:\Users\MikeB\Desktop\Academy\FAU\Semester 4\ABA\Dataset\Global_Terminology_Dictionary.json"

evaluation_dataset = []
global_terminology_dictionary = {}

print("Starting LLM automated dataset generation...")

# --- THIS IS THE FIX: os.walk() searches all subfolders ---
for root, dirs, files in os.walk(RAW_DIR):
    for filename in files:
        if filename.endswith(".pdf"):
            filepath = os.path.join(root, filename)
            print(f"Processing: {filename}")
            
            loader = PyPDFLoader(filepath)
            pages = loader.load_and_split()
            
            for i, page in enumerate(pages):
                if len(page.page_content) < 100: 
                    continue
                    
                try:
                    result = chain.invoke({"text": page.page_content})
                    
                    evaluation_dataset.append({
                        "Source_File": filename,
                        "Page_Number": i + 1,
                        "Formal_Context": page.page_content.strip(),
                        "Formal_Query": result.get("formal_query", ""),
                        "Colloquial_Query": result.get("colloquial_query", "")
                    })
                    
                    for mapping in result.get("mappings", []):
                        informal = mapping.get("informal", "").lower()
                        formal = mapping.get("formal", "").lower()
                        if informal and formal:
                            global_terminology_dictionary[informal] = formal

                except Exception as e:
                    print(f"Error processing page {i+1} of {filename}: {e}")

df = pd.DataFrame(evaluation_dataset)
df.to_csv(CSV_OUT, index=False)

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(global_terminology_dictionary, f, indent=4)

print("Pipeline complete! Files saved in your Dataset folder.")
