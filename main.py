from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from cerebras.frameworks.langchain import CerebrasLLM  # Cerebras LLM integration

# Configure environment variables for API keys and Neo4j credentials
import os
os.environ["NEO4J_URI"] = "neo4j+s://<your_neo4j_uri>"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "<**********************>"
os.environ["CEREBRAS_API_KEY"] = "<**********************>"

# Initialize Neo4j Graph
graph = Neo4jGraph()

# Load Wikipedia data (for example purposes, we use Elizabeth I)
from langchain.document_loaders import WikipediaLoader
raw_documents = WikipediaLoader(query="Elizabeth I").load()

# Configure CerebrasLLM as the language model for inference
llm = CerebrasLLM(api_key=os.environ["CEREBRAS_API_KEY"])

# Embedding setup with Neo4jVector for unstructured data storage
vector_index = Neo4jVector.from_existing_graph(
    llm.embeddings(), 
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
)

# Define Entity Extraction Prompt
class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(..., description="Entity names in the text")

entity_prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract entities from the text"),
    ("human", "{question}")
])

entity_chain = entity_prompt | llm.with_structured_output(Entities)

# Full-Text Search Query
def generate_full_text_query(input_text: str) -> str:
    full_text_query = " AND ".join([f"{word}~2" for word in input_text.split()])
    return full_text_query

# Structured Retriever Function
def structured_retriever(question: str) -> str:
    entities = entity_chain.invoke({"question": question}).names
    response = graph.query(f"MATCH (e)-[r:MENTIONS]->(related) WHERE e.id IN {entities}")
    return "\n".join([f"{r['source']} - {r['type']} -> {r['target']}" for r in response])

# Unified Retriever
def retriever(question: str):
    structured_data = structured_retriever(question)
    unstructured_data = [doc.page_content for doc in vector_index.similarity_search(question)]
    return f"Structured Data:\n{structured_data}\nUnstructured Data:\n{unstructured_data}"

# Conversational Flow with Follow-Up Management
CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_template("""
Given the conversation history and follow-up question, rephrase the question to be standalone.
Chat History: {chat_history}
Follow Up Input: {question}
Standalone Question:
""")

def condense_question(chat_history, question):
    return CONDENSE_QUESTION_PROMPT.invoke({"chat_history": chat_history, "question": question})

# Main Query Chain with Cerebras LLM
def main_query(question: str, chat_history: List[Tuple[str, str]] = []):
    if chat_history:
        standalone_question = condense_question(chat_history, question)
    else:
        standalone_question = question
    
    context = retriever(standalone_question)
    return llm.invoke({"context": context, "question": standalone_question})

# Sample Queries
print(main_query("Which house did Elizabeth I belong to?"))
