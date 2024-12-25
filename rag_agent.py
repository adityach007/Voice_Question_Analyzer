from langchain.agents import AgentExecutor, Tool, initialize_agent, AgentType
from langchain_groq import ChatGroq
from langchain.tools import DuckDuckGoSearchRun
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, create_extraction_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any
from pathlib import Path
import logging
import json
import time

# Update logging configuration for better visibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to terminal
        logging.FileHandler('rag_agent.log')  # Also save to file
    ]
)

logger = logging.getLogger(__name__)

class RetrievalStrategy:
    def __init__(self, name: str, description: str, score: float = 0.5):
        self.name = name
        self.description = description
        self.score = score
        self.usage_count = 0
        self.success_count = 0
    
    def update_score(self, success: bool) -> None:
        try:
            self.usage_count += 1
            if success:
                self.success_count += 1
            self.score = self.success_count / self.usage_count
        except Exception as e:
            logger.error(f"Error updating strategy score: {e}")
            self.score = 0.5

class DocumentStore:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_stores = {}
        self.setup_vector_stores()
    
    def setup_vector_stores(self):
        try:
            current_dir = Path(__file__).parent.absolute()
            logger.info(f"Current directory: {current_dir}")
            
            # Setup automotive document only
            auto_pdf_path = current_dir / "automotive.pdf"
            self._verify_and_load_document(auto_pdf_path, "automotive")
            
            # Verify vector store was created
            logger.info(f"Vector stores created: {list(self.vector_stores.keys())}")
            for doc_type, store in self.vector_stores.items():
                if hasattr(store, 'index'):
                    logger.info(f"{doc_type} vector store size: {store.index.ntotal}")
                
        except Exception as e:
            logger.error(f"Error setting up vector stores: {str(e)}")
            raise

    def _verify_and_load_document(self, pdf_path: Path, doc_type: str):
        logger.info(f"Attempting to load {doc_type} from: {pdf_path}")
        if not pdf_path.exists():
            logger.error(f"File not found: {pdf_path}")
            raise FileNotFoundError(f"Could not find {doc_type} PDF at {pdf_path}")
        if pdf_path.stat().st_size == 0:
            logger.error(f"File is empty: {pdf_path}")
            raise ValueError(f"{doc_type} PDF is empty")
        self._load_document(pdf_path, doc_type)

    def _load_document(self, pdf_path: Path, doc_type: str):
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()
            
            if not documents:
                logger.error(f"No content loaded from {doc_type} PDF")
                raise ValueError(f"No content could be extracted from {doc_type} PDF")
            
            logger.info(f"Loaded {len(documents)} pages from {doc_type}")
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
            splits = text_splitter.split_documents(documents)
            
            if not splits:
                logger.error(f"No splits created for {doc_type}")
                raise ValueError(f"Text splitting failed for {doc_type}")
            
            logger.info(f"Created {len(splits)} splits for {doc_type}")
            
            self.vector_stores[doc_type] = FAISS.from_documents(splits, self.embeddings)
            logger.info(f"Vector store for {doc_type} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading {doc_type} document: {e}")
            raise

class StrategyCoordinator:
    def __init__(self, llm):
        self.llm = llm
        self.strategies = {
            "semantic": RetrievalStrategy("semantic", "Dense vector similarity search"),
            "keyword": RetrievalStrategy("keyword", "Keyword-based search"),
            "hybrid": RetrievalStrategy("hybrid", "Combined semantic and keyword search")
        }
    
    def select_strategy(self, query: str, context: Dict[str, Any]) -> str:
        try:
            # Logic to select best strategy
            prompt = PromptTemplate(
                template="""Analyze the query and context to select the best search strategy.
                Output exactly one word: 'semantic', 'keyword', or 'hybrid'.
                Query: {query}
                Context: {context}
                Choice:""",
                input_variables=["query", "context"]
            )
            
            chain = LLMChain(llm=self.llm, prompt=prompt)
            strategy = chain.run(query=query, context=str(context)).strip().lower()
            return strategy if strategy in self.strategies else "hybrid"
        except Exception as e:
            logger.error(f"Strategy selection error: {e}")
            return "hybrid"

class RAGAgent:
    def __init__(self):
        self.llm = ChatGroq(
            api_key="YOUR_GROQ_API_KEY",
            model_name="mixtral-8x7b-32768",
            temperature=0.7
        )
        self.doc_store = DocumentStore()
        self.search = DuckDuckGoSearchRun(
            max_retries=3,  # Add retries for rate limiting
            sleep_time=2    # Add delay between retries
        )
        self.strategy_coordinator = StrategyCoordinator(self.llm)
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        self.tools = self._create_tools()
        self.agent = None  # Will be initialized later
        self.agent_executor = None  # Will be initialized later
        self._initialize_components()
        self.automotive_terms = {
            'vehicle_specs': ['suv', 'car', 'vehicle', 'specification', 'dimensions', 'measurements', 'weight', 'capacity'],
            'safety': ['safety', 'airbag', 'advanced driver', 'adas', 'brake', 'collision', 'warning'],
            'features': ['feature', 'technology', 'system', 'infotainment', 'entertainment', 'connectivity'],
            'maintenance': ['maintenance', 'service', 'warranty', 'repair', 'insurance', 'schedule'],
            'performance': ['engine', 'power', 'torque', 'transmission', 'speed', 'acceleration']
        }
        
    def _initialize_components(self):
        """Initialize agent components in the correct order"""
        try:
            # Initialize tools first
            self.tools = self._create_tools()
            
            # Initialize agent with proper configuration
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
                verbose=True,
                max_iterations=3,
                early_stopping_method="generate",
                memory=self.memory,
            )
            
            # Initialize agent executor
            self.agent_executor = AgentExecutor.from_agent_and_tools(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=3,
                handle_parsing_errors=True,
                return_intermediate_steps=False  # Added this to fix the plan() error
            )
            
            logger.info("Agent components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agent components: {e}")
            raise
    
    def _create_tools(self):
        return [
            Tool(
                name="Vector_Search",
                func=self._vector_search,
                description="Search through document embeddings for document-related queries."
            ),
            Tool(
                name="Web_Search",
                func=self.search.run,
                description="Search the web for general information or current topics."
            )
        ]
    
    def _vector_search(self, query: str, doc_type: str = None) -> str:
        try:
            if not doc_type:
                _, doc_type = self._should_use_vector_search(query)
            
            vector_store = self.doc_store.vector_stores.get(doc_type)
            if vector_store:
                logger.info(f"Performing {doc_type} vector search for: {query}")
                
                # Get more targeted results with score
                results_with_scores = vector_store.similarity_search_with_score(query, k=3)
                
                if results_with_scores:
                    # Filter results with similarity threshold
                    relevant_results = [
                        (doc, score) for doc, score in results_with_scores 
                        if score < 1.0  # Lower score means better match
                    ]
                    
                    if relevant_results:
                        # Sort by score and format results
                        relevant_results.sort(key=lambda x: x[1])  # Sort by similarity score
                        
                        # Extract and format the most relevant content
                        formatted_response = ""
                        for doc, score in relevant_results:
                            content = doc.page_content.strip()
                            # Add section markers if they exist in the document metadata
                            section = doc.metadata.get('section', '')
                            if section:
                                formatted_response += f"\n[{section}]: {content}"
                            else:
                                formatted_response += f"\n{content}"
                        
                        logger.info(f"Found {len(relevant_results)} relevant matches")
                        return formatted_response.strip()
                        
                logger.info("No relevant matches found in vector search")
                return f"No relevant information found in {doc_type} documentation."
                
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return "Error performing vector search."

    def _validate_question(self, question: str) -> str:
        """Validate and truncate question if needed"""
        words = question.split()
        if len(words) > 50:
            truncated = ' '.join(words[:50])
            logger.info(f"Truncated question from {len(words)} to 50 words")
            return truncated + "..."
        return question

    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata and questions from text with improved logging"""
        try:
            print("\n" + "="*50)
            print("EXTRACTING QUESTIONS FROM SPEECH")
            print("="*50)
            print(f"Analyzing text: {text[:200]}...")

            schema = {
                "properties": {
                    "questions": {
                        "type": "array", 
                        "items": {
                            "type": "string",
                            "maxLength": 250  # Approximately 50 words
                        }
                    },
                    "topics": {"type": "array", "items": {"type": "string"}},
                    "complexity": {"type": "string", "enum": ["simple", "moderate", "complex"]}
                },
                "required": ["questions", "topics", "complexity"]
            }
            
            extraction_prompt = PromptTemplate(
                template="""Extract questions from the following text. Include both explicit and implicit questions.
                Format each question clearly and accurately.
                Important: Each question must be 50 words or less.
                
                Text: {text}
                
                Instructions:
                1. Identify direct questions
                2. Identify implied questions
                3. Format as complete, well-formed questions
                4. Ensure each question is concise (≤50 words)
                
                Output in JSON format with 'questions' array.""",
                input_variables=["text"]
            )
            
            chain = create_extraction_chain(schema, self.llm)
            result = chain.run(text)
            
            if result and isinstance(result, list) and len(result) > 0:
                # Validate and truncate questions if needed
                validated_questions = [
                    self._validate_question(q) 
                    for q in result[0].get('questions', [])
                ]
                
                result[0]['questions'] = validated_questions
                
                print("\nExtracted Questions:")
                for idx, q in enumerate(validated_questions):
                    print(f"{idx + 1}. {q}")
                return result[0]
            else:
                print("No questions extracted!")
                return {
                    "questions": [],
                    "topics": [],
                    "complexity": "moderate"
                }
        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            print(f"Error extracting questions: {e}")
            return {"questions": [], "topics": [], "complexity": "moderate"}

    def _initialize_agent(self):
        PREFIX = """You are an intelligent assistant that can analyze speech and identify questions.
        For document-related queries about vehicles (especially the 2024 Model X SUV), use Vector_Search first.
        For general queries or if Vector_Search yields no results, use Web_Search.
        Always provide detailed, accurate responses and cite sources when possible.
        
        When using Vector_Search:
        1. Look for specific vehicle information in the document
        2. Extract relevant details about features, specifications, or safety
        3. If found, use that information as the primary source
        
        When using Web_Search:
        1. Search for current and general information
        2. Focus on relevant details to the query
        3. Provide well-structured, informative responses
        
        Always explain your thought process and which tool you're using."""
        
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=self.memory,
            agent_kwargs={
                "prefix": PREFIX,
                "verbose": True
            }
        )

    def _should_use_vector_search(self, query: str) -> tuple[bool, str]:
        """Determine if query should use vector search and which context"""
        query_lower = query.lower()
        
        # Check automotive terms
        for category, terms in self.automotive_terms.items():
            if any(term in query_lower for term in terms):
                return True, "automotive"
        
        return False, "web"

    def _setup_agent_executor(self):
        """Set up agent executor with custom prompt and tools"""
        prompt = PromptTemplate(
            template="""Process this query using the appropriate search method.
            For automotive-related queries about specifications, safety, features, or maintenance, 
            use Vector_Search first. If no relevant results are found, then use Web_Search.
            For general queries, use Web_Search directly.
            
            Current query: {query}
            Thought process:
            1. Analyze if query is automotive-related
            2. Choose appropriate search method
            3. Process results
            4. Provide detailed response
            
            {format_instructions}
            """,
            input_variables=["query"],
            partial_variables={
                "format_instructions": "Format your response clearly with source attribution."
            }
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3
        )

    def process_query(self, query: str, metadata: Dict[str, Any]) -> str:
        """Process a single query using agent executor"""
        try:
            print("\n" + "="*50)
            print(f"Processing Query: {query}")
            print("="*50)
            
            use_vector, doc_type = self._should_use_vector_search(query)
            print(f"Using vector search: {use_vector} for {doc_type}")
            
            if use_vector:
                print("Attempting automotive vector search...")
                focused_query = self._create_focused_query(query, "automotive")
                print(f"Focused query: {focused_query}")
                
                result = self._vector_search(focused_query, "automotive")
                if result and "No relevant information found" not in result:
                    print("Vector search successful!")
                    return f"[From Vehicle Documentation] {result}"
                print("Vector search yielded no results, falling back to web search...")
            
            # Try web search with retry mechanism
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    web_result = self.search.run(query)
                    if web_result:
                        return f"[From Web Search] {web_result}"
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Web search attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print("All web search attempts failed")
                        return "I apologize, but I'm currently unable to access the information due to rate limits. Please try again in a few moments."
            
        except Exception as e:
            print(f"\nError in query processing: {e}")
            logger.error(f"Error in query processing: {e}")
            return f"Error processing query: {str(e)}"

    def _create_focused_query(self, query: str, doc_type: str) -> str:
        """Create a more focused query based on the document type and question"""
        query_lower = query.lower()
        
        if doc_type == "automotive":
            # Map question types to automotive sections
            for category, terms in self.automotive_terms.items():
                if any(term in query_lower for term in terms):
                    return f"{category}: {query}"
        
        return query

def process_audio_input(audio_data):
    """Process audio input with improved visibility"""
    try:
        print("\n" + "="*50)
        print("STARTING AUDIO PROCESSING")
        print("="*50)
        
        rag_agent = RAGAgent()
        
        # Transcribe audio
        from audio_handler import transcribe_audio
        transcription = transcribe_audio(audio_data)
        print(f"\nTranscription:\n{transcription}")
        
        # Extract questions
        metadata = rag_agent.extract_metadata(transcription)
        questions = metadata.get("questions", [])
        
        if not questions:
            print("No questions detected in the speech!")
            return {
                "transcription": transcription,
                "questions": None,
                "answers": None
            }
        
        print("\n" + "="*50)
        print("PROCESSING QUESTIONS")
        print("="*50)
        
        try:
            answers = []
            for idx, question in enumerate(questions, 1):
                print(f"\nProcessing Question {idx}: {question}")
                answer = rag_agent.process_query(question, metadata)
                print(f"Answer {idx}: {answer}")
                answers.append(f"Question {idx}: {question}\nAnswer: {answer}")
            
            formatted_answers = "\n\n".join(answers)
            
        except Exception as e:
            error_msg = f"Error processing questions: {str(e)}"
            print(error_msg)
            formatted_answers = error_msg
        
        return {
            "transcription": transcription,
            "questions": questions,
            "answers": formatted_answers
        }
        
    except Exception as e:
        error_msg = f"Error in process_audio_input: {e}"
        logger.error(error_msg)
        print(error_msg)
        return {
            "transcription": "Error processing audio",
            "questions": None,
            "answers": None
        }
