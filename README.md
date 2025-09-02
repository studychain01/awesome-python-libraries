# Awesome Python Libraries & Tools

A curated collection of powerful Python libraries and tools organized by use case, perfect for developers, data scientists, and AI enthusiasts.

## 🤖 AI & Machine Learning Frameworks

### LangChain & RAG (Retrieval-Augmented Generation)
- **LangChain** - Framework for building LLM applications with chunking strategies
  - **Header-based chunking** - Break documents by headers for better context
  - **Sentence-based chunking** - Break documents by sentences for granular processing
  - **Use case**: Document processing, content summarization, Q&A systems
  - **Install**: `pip install langchain langchain-openai langchain-core langchain_community`

- **Chroma** - Vector database for AI applications
  - **Embedding storage** - Store and query vector embeddings
  - **LangChain integration** - Seamless integration with LangChain workflows
  - **Local deployment** - Run locally or in the cloud
  - **Use case**: RAG applications, semantic search, document retrieval
  - **Install**: `pip install langchain_chroma`

- **Sentence Transformers** - State-of-the-art sentence embeddings
  - **Pre-trained models** - High-quality embeddings for 100+ languages
  - **Easy integration** - Simple API for text-to-vector conversion
  - **Multiple models** - BERT, RoBERTa, DistilBERT, and more
  - **Use case**: Semantic search, text similarity, RAG applications
  - **Install**: `pip install sentence_transformers`

### Vector Databases & Embeddings
- **Weaviate** (weaviate.io) - Vector database for AI applications
  - **Embedding models** - Multiple embedding model providers
  - **Ranking algorithms** - Advanced similarity search and ranking
  - **Use case**: Semantic search, recommendation systems, AI-powered applications
  - **Docs**: https://docs.weaviate.io/weaviate/model-providers/weaviate/embeddings

- **Deep Lake** (deeplake.ai) - Vector database for AI applications
  - **Data lake for AI** - Store and query vectors, images, videos, and documents
  - **Version control** - Git-like versioning for datasets
  - **Use case**: Computer vision, multimodal AI, dataset management
  - **Website**: https://www.deeplake.ai/

- **Zilliz Cloud** - Managed Milvus vector database service
  - **Managed Milvus** - Fully managed vector database in the cloud
  - **High performance** - Optimized for large-scale vector operations
  - **Auto-scaling** - Automatic scaling based on workload
  - **Use case**: Production vector search, AI applications, similarity search
  - **Website**: https://zilliz.com/cloud

- **ActiveLoop** - Platform for building AI applications with Deep Lake
  - **Integration** - Seamless Deep Lake integration
  - **Use case**: AI application development, data pipeline management
  - **Website**: https://www.activeloop.ai/

### Graph Databases
- **Neo4j** - Graph database for relationship-heavy data
  - **GraphAcademy** - Learning platform for Neo4j (graphacademy.neo4j.com)
  - **Certifications** - Professional Neo4j certifications available
  - **Use case**: Social networks, recommendation engines, fraud detection
  - **Website**: https://neo4j.com/

### NLP & Text Processing
#### Document Processing
- **docx2txt** - Extract text from Microsoft Word documents
  - **Word document parsing** - Convert .docx files to plain text
  - **Simple API** - Easy-to-use text extraction
  - **No dependencies** - Lightweight library with minimal requirements
  - **Use case**: Document text extraction, content processing, RAG applications
  - **Install**: `pip install docx2txt`

- **PyPDF** - PDF processing and text extraction
  - **PDF text extraction** - Extract text from PDF documents
  - **PDF manipulation** - Create, modify, and merge PDFs
  - **Metadata handling** - Access PDF metadata and properties
  - **Use case**: PDF processing, document analysis, content extraction
  - **Install**: `pip install pypdf`

- **html2text** - Convert HTML to clean, readable text
  - **HTML parsing** - Convert HTML documents to plain text or Markdown
  - **Clean formatting** - Preserves structure while removing HTML tags
  - **Markdown output** - Option to convert HTML to Markdown format
  - **Use case**: Web scraping, content extraction, HTML processing, RAG applications
  - **Install**: `pip install html2text`
  - **Usage**: `from html2text import html2text`

#### Named Entity Recognition (NER)
- **spaCy** - Industrial-strength NLP library
  - **Pre-trained models** - High-accuracy NER models for multiple languages
  - **Custom training** - Train custom NER models for domain-specific entities
  - **Use case**: Information extraction, document processing, data cleaning
  - **Website**: https://spacy.io/

- **Hugging Face Transformers** - State-of-the-art transformer models
  - **BERT, RoBERTa, DistilBERT** - Pre-trained models for NER
  - **Easy fine-tuning** - Simple API for custom NER models
  - **Use case**: Advanced NLP tasks, research, production applications
  - **Website**: https://huggingface.co/

- **Stanza** (by Stanford) - Multi-language NLP toolkit
  - **Multi-language support** - 70+ languages supported
  - **Stanford quality** - Research-grade NLP models
  - **Use case**: Multi-language applications, research, academic projects
  - **Website**: https://stanfordnlp.github.io/stanza/

- **Flair** - Simple framework for state-of-the-art NLP
  - **Zero-shot learning** - No training data required for many tasks
  - **Contextual embeddings** - Advanced embedding techniques
  - **Use case**: Quick prototyping, research, production NLP
  - **Website**: https://github.com/flairNLP/flair

- **OpenAI/GPT-4/Claude** - Large language model APIs
  - **Zero-shot NER** - Extract entities without training
  - **Flexible prompting** - Custom entity extraction strategies
  - **Use case**: Quick prototyping, flexible entity extraction
  - **APIs**: OpenAI, Anthropic Claude

#### Relationship Extraction
- **OpenIE (Stanford)** - Open Information Extraction
  - **Open-domain extraction** - Extract relationships from any text
  - **Triple extraction** - Subject-predicate-object relationships
  - **Use case**: Knowledge graph construction, information extraction
  - **Website**: https://nlp.stanford.edu/software/openie.html

- **spaCy + Dependency Parsing** - Rule-based relationship extraction
  - **Linguistic patterns** - Grammar-based relationship extraction
  - **Custom rules** - Domain-specific relationship patterns
  - **Use case**: Structured information extraction, domain-specific applications

- **REBEL** - Relation extraction with BERT
  - **BERT-based extraction** - State-of-the-art relation extraction
  - **Pre-trained models** - Ready-to-use relation extraction
  - **Use case**: Knowledge graph construction, research
  - **GitHub**: https://github.com/Babelscape/rebel

- **KAIROS** - Temporal and causal relationship extraction
  - **Temporal reasoning** - Extract time-based relationships
  - **Causal inference** - Identify cause-effect relationships
  - **Use case**: Event analysis, temporal reasoning, causal discovery
  - **Website**: https://github.com/isi-nlp/Kairos

- **DyGIE++** - Dynamic Graph Information Extraction
  - **Joint extraction** - Extract entities and relations together
  - **Graph-based approach** - Dynamic graph information extraction
  - **Use case**: Complex information extraction, research
  - **GitHub**: https://github.com/luanyi/DyGIE

## 🔒 Safety & Guardrails Frameworks

### Output Validation & Structure Enforcement
- **Guardrails.ai** - Enforces output schema and validates LLM responses
  - **JSON validation** - Ensures LLM outputs match expected JSON structure
  - **Custom validators** - Create domain-specific validation rules
  - **Retry logic** - Automatically retries on invalid responses
  - **Use case**: Production LLM applications, API response validation
  - **Website**: https://www.guardrailsai.com/

- **Outlines** - Light-weight structured output enforcement
  - **Python typing** - Uses Python type hints for validation
  - **Simple integration** - Easy to add to existing LLM pipelines
  - **Use case**: Quick validation, prototyping, structured outputs
  - **GitHub**: https://github.com/outlines-dev/outlines
  - **GitHub**: https://github.com/dottxt-ai/outlines?tab=readme-ov-file
  

- **Pydantic + LLM** - Validates LLM output using Python dataclass models
  - **Type safety** - Leverages Pydantic for robust validation
  - **Custom models** - Define expected output structure
  - **Use case**: Type-safe LLM applications, data validation
  - **Documentation**: https://docs.pydantic.dev/

### Content Moderation & Safety Filtering
- **Rebuff** - Real-time moderation of LLM responses
  - **Toxicity detection** - Identifies harmful or inappropriate content
  - **Hallucination detection** - Detects false or fabricated information
  - **Bias detection** - Identifies biased or unfair content
  - **Use case**: Content moderation, safe AI deployment
  - **Website**: https://rebuff.ai/

- **Llama Guard (Meta)** - Classifies LLM input/output for harmful content
  - **Multi-classification** - Detects various types of harmful content
  - **Open source** - Free and customizable
  - **Production ready** - Used by Meta for content moderation
  - **Use case**: Content filtering, safety compliance
  - **GitHub**: https://github.com/meta-llama/PurpleLlama

- **HarmBench (CAIS)** - Testing framework for LLM safety evaluation (like pytest for AI)
  - **Comprehensive testing** - Tests multiple safety dimensions and harmful behaviors
  - **Red teaming methods** - 18+ automated red teaming methods included
  - **Model support** - 33+ target LLMs and defenses evaluated
  - **Testing framework** - Standardized evaluation pipeline for AI safety testing
  - **Use case**: LLM testing, safety evaluation, red teaming, automated testing
  - **Website**: https://www.harmbench.org/
  - **GitHub**: https://github.com/centerforaisafety/HarmBench

### Evaluation & Red Teaming
- **RAGAS** - Testing framework for RAG systems (like pytest for RAG pipelines)
  - **RAG-specific metrics** - Designed for retrieval-augmented generation evaluation
  - **Multiple dimensions** - Faithfulness, fluency, relevance, context precision
  - **Synthetic data generation** - Creates high-quality evaluation datasets
  - **Online monitoring** - Production monitoring and quality assessment
  - **Easy integration** - Works with popular RAG frameworks (LangChain, LlamaIndex)
  - **Use case**: RAG system testing, quality assessment, automated evaluation
  - **Website**: https://www.ragas.io/
  - **GitHub**: https://github.com/explodinggradients/ragas

- **Helm (Stanford CRFM)** - Benchmarking framework for LLM evaluation (like pytest for benchmarking)
  - **Comprehensive benchmarks** - Multiple evaluation dimensions
  - **Academic rigor** - Research-grade evaluation framework
  - **Open source** - Free and extensible
  - **Benchmarking framework** - Standardized evaluation pipeline for LLM benchmarking
  - **Use case**: Model evaluation, research, benchmarking, automated testing
  - **Website**: https://crfm.stanford.edu/helm/

- **OpenAI Eval Harness** - Unit test style framework for LLMs (like pytest for LLM testing)
  - **Unit-style tests** - Run unit tests on LLM outputs with expected answers
  - **Custom grading logic** - Define expected answers and grading criteria
  - **Task-specific evals** - SQL generation, chain-of-thought reasoning, bias detection
  - **Safety testing** - Style checks, safety refusals, custom safety tests
  - **Automated testing** - Run tests across models or prompt versions
  - **Use case**: LLM unit testing, custom evaluation, automated testing
  - **GitHub**: https://github.com/openai/evals

### Prompt Injection Protection
- **Llama Prompt Guard 2 (Meta)** - Detects prompt injection and jailbreaking attacks
  - **Prevention** - Blocks malicious prompts before processing
  - **Prompt Injection Checker** - Blocks bad prompts
  - **Real-time detection** - Analyzes prompts for injection patterns
  - **Multiple model sizes** - 86M (multilingual) and 22M (English-focused) versions
  - **Llama integration** - Designed for Llama 3 and Llama 4 models
  - **Use case**: Security, prompt injection prevention
  - **Documentation**: https://www.llama.com/docs/model-cards-and-prompt-formats/prompt-guard/


### Prompt Management & Safety
- **Promptlayer** - Prompt management and evaluation platform
  - **Prompt versioning** - Track and manage prompt changes
  - **Evaluation tools** - Built-in safety and quality evaluation
  - **Team collaboration** - Share and collaborate on prompts
  - **Use case**: Prompt engineering, team collaboration, safety tracking
  - **Website**: https://promptlayer.com/

## 🔍 Search & Research Tools

### Web Search APIs
- **Tavily** (tavily.com) - API for internet searches with LangChain integration
  - **Use case**: Real-time web research, fact-checking, content discovery
  - **Integration**: Works seamlessly with LangChain for AI-powered search

- **Exa** (exa.ai) - Web search API built specifically for LLMs
  - **Neural ranking** - AI-powered search results ranking
  - **Multiple endpoints** - Search, Answer, Research, Websets
  - **Enterprise features** - SOC2 certified, zero data retention
  - **Use case**: AI-powered search, research automation, enterprise LLM applications
  - **Website**: https://exa.ai/

- **Browserbase** - Browser automation and web scraping API
  - **Browser automation** - Control real browsers for complex web interactions
  - **Web scraping** - Extract data from dynamic websites and SPAs
  - **JavaScript rendering** - Handle JavaScript-heavy sites and modern web apps
  - **Use case**: Web scraping, browser automation, data extraction
  - **Website**: https://browserbase.com/

### Web Scraping
- **Firecrawl** (firecrawl.dev) - Web scraping tool
  - **Use case**: Data extraction, content monitoring, research automation

### Business Intelligence
- **BBB (Better Business Bureau)** (bbb.org) - Business verification and ratings
  - **Use case**: Business research, vendor verification, trust assessment

- **Glean** (glean.com) - Company search and business intelligence platform
  - **Company database** - Comprehensive database of businesses
  - **Business research** - Find companies by industry, size, location
  - **Contact information** - Business contact details and profiles
  - **Use case**: Sales prospecting, market research, competitive analysis
  - **Website**: https://glean.com/

## 🏗️ Agent Frameworks & MCP (Model Context Protocol)

### Agent Development
- **Modus Agent Framework** (hypermode.com) - Framework for building AI agents
  - **GitHub**: https://github.com/hypermodeinc/modus
  - **Documentation**: https://docs.hypermode.com/modus/first-modus-agent
  - **Use case**: Custom AI agents, workflow automation, task-specific AI

- **Mastra** - TypeScript Agent Framework for AI features
  - **Modern JavaScript stack** - Built for prototyping and productionizing AI features
  - **Agent orchestration** - Create flexible multi-agent architectures
  - **Workflow management** - Durable graph-based state machines with tracing
  - **Unified RAG** - Vector store integration with metadata filtering
  - **Observability** - Performance metrics, evals, and tracing
  - **Use case**: AI agent development, workflow automation, TypeScript AI applications
  - **Website**: https://mastra.ai/

### Voice AI & Communication
- **Vapi** - Voice AI agents for developers
  - **Voice AI platform** - Most configurable API for building voice AI products
  - **Inbound/Outbound calls** - Handle millions of calls with enterprise reliability
  - **Multilingual support** - 100+ languages including English, Spanish, Mandarin
  - **Tool calling** - Plug in APIs as tools for intelligent data fetching
  - **Enterprise features** - 99.99% uptime, SOC2/HIPAA/PCI compliant
  - **40+ integrations** - OpenAI, Anthropic, Twilio, Salesforce, and more
  - **Use case**: Voice AI applications, call center automation, phone operations
  - **Website**: https://vapi.ai/

### Personal Data & Context Management
- **Basic** - Universal context for LLMs with Personal Data Stores
  - **No-DB, no-auth design** - Simple data storage without complex infrastructure
  - **Personal Data Stores** - Access user context across applications
  - **Open federated protocol** - Data is everlasting and portable
  - **AI app memory layer** - Store and access user context from previous interactions
  - **Enterprise support** - Teams from Cornell, YC, Browserbase, Lemniscap, EYP
  - **Use case**: AI app memory, user context management, personal data storage
  - **Website**: https://basic.tech/

### MCP Servers
- **Magic MCP** - Popular MCP server implementation
- **Use case**: Model context protocol implementations, AI tool integration

### Workflow Automation
- **Pipedream** - Workflow automation and API integration platform
  - **Event-driven workflows** - Connect APIs and services automatically
  - **Python support** - Write custom Python code in workflows
  - **Use case**: API integrations, data pipelines, automation workflows
  - **Website**: https://pipedream.com/

- **n8n** - Open-source workflow automation platform
  - **Visual workflow builder** - Drag-and-drop interface for workflows
  - **Self-hosted option** - Deploy on your own infrastructure
  - **Extensive integrations** - 200+ nodes for different services
  - **Use case**: Complex workflow automation, enterprise integrations
  - **Website**: https://n8n.io/

- **Composio** - AI-native workflow automation platform
  - **AI-powered workflows** - Natural language workflow creation
  - **No-code automation** - Build complex workflows without coding
  - **Enterprise integrations** - Connect to business systems and APIs
  - **Use case**: AI workflow automation, business process automation
  - **Website**: https://composio.dev/

- **Sim** - Visual AI workflow editor for building AI-powered applications
  - **Visual workflow editor** - Drag and drop components on canvas without code
  - **Multi-model AI support** - Connect to OpenAI, Anthropic, Google, Groq, Cerebras, Ollama
  - **60+ pre-built tools** - Gmail, Slack, Notion, Google Sheets, Airtable, Supabase, Pinecone
  - **Flexible execution** - Run via chat, REST API, webhooks, scheduled jobs, or external triggers
  - **Real-time collaboration** - Work simultaneously with team members like Google Docs
  - **Production deployment** - Deploy as APIs, integrate with SDK, or embed as plugins
  - **Use case**: AI assistants, content generation, data processing, process automation, API orchestration
  - **Website**: https://www.sim.ai/

- **Apify** - Web scraping and automation platform
  - **Web scraping tools** - Extract data from websites at scale
  - **Automation platform** - Build and deploy web automation
  - **Cloud infrastructure** - Scalable scraping and automation
  - **Use case**: Web scraping, data extraction, automation
  - **Website**: https://apify.com/

- **Stagehand ** - AI-native browser automation framework
  - **AI-driven workflows** - Natural language browser control
  - **Playwright compatibility** - Works with existing Playwright scripts
  - **Self-healing automation** - Adapts to page changes automatically
  - **Deterministic automation** - Atomic steps for reliable execution
  - **Use case**: Browser automation, AI agents, web scraping
  - **Website**: https://www.stagehand.dev/

- **Browser Use** - AI browser agent for repetitive tasks
  - **No-code automation** - Automate repetitive online tasks without coding
  - **Data scraping** - Extract data from complex websites with AI
  - **Form filling** - Automatically fill out online forms
  - **Visual interface** - Chat-based task execution
  - **Open source** - Largest open source project for browser agents
  - **Use case**: Browser automation, data entry, form filling, web scraping
  - **Website**: https://browser-use.com/

## 🎯 Practical Use Cases & Applications

### Travel & Recommendation Systems
- **Travel Itinerary Planner Agent** - AI-powered travel planning
  - **Use case**: Personalized travel recommendations, itinerary optimization
  - **Example**: "I want to go to this type of event or place" → LLM suggests best options

### Content Processing
- **Web Scraping Agent** - Automated website summarization
  - **Use case**: Quick content digestion, research acceleration
  - **Example**: "I have no time to read, summarize this website for me"

### Term Graph & Vector Similarity
- **Term Graph RAG** - Vector similarity search for terms
  - **Use case**: Semantic search, concept mapping, knowledge discovery

### Information Extraction
- **Entity Recognition Pipeline** - Extract entities from documents
  - **Use case**: Document processing, data extraction, knowledge graphs
  - **Tools**: spaCy, Hugging Face, Flair, LLMs

- **Relationship Extraction Pipeline** - Extract relationships between entities
  - **Use case**: Knowledge graph construction, information extraction
  - **Tools**: OpenIE, REBEL, KAIROS, DyGIE++

## 🚀 Getting Started

### For Beginners
1. Start with **LangChain** for basic LLM applications
2. Explore **Weaviate**, **Deep Lake**, or **Zilliz Cloud** for vector database needs
3. Try **Tavily** or **Exa** for web search integration
4. Use **spaCy** for basic NLP tasks

### For Advanced Users
1. Build custom agents with **Modus Framework**
2. Implement graph databases with **Neo4j**
3. Create specialized MCP servers
4. Automate workflows with **Pipedream** or **n8n**
5. Train custom NER models with **Hugging Face Transformers**
6. Add safety guardrails with **Guardrails.ai** or **Rebuff**
7. Evaluate RAG systems with **RAGAS**

### For Business Applications
1. Use **BBB API** for business verification
2. Research companies with **Glean**
3. Build custom agents with **Modus Framework**
4. Extract business entities with **spaCy** or **Flair**
5. Implement content moderation with **Llama Guard** or **Rebuff**
6. Add prompt injection protection with **Llama Prompt Guard 2**

### For NLP Research
1. Use **Stanza** for multi-language NLP
2. Implement **REBEL** for relation extraction
3. Explore **KAIROS** for temporal reasoning
4. Build knowledge graphs with **OpenIE** and **Neo4j**

## 📚 Learning Resources

- **Neo4j GraphAcademy**: https://graphacademy.neo4j.com/
- **Weaviate Documentation**: https://docs.weaviate.io/
- **Modus Framework**: https://docs.hypermode.com/modus/first-modus-agent
- **Tavily API**: https://www.tavily.com/
- **spaCy Tutorials**: https://spacy.io/usage
- **Hugging Face Course**: https://huggingface.co/course

## 🔗 Quick Reference Links

- [Weaviate](https://weaviate.io/)
- [Deep Lake](https://www.deeplake.ai/)
- [Zilliz Cloud](https://zilliz.com/cloud)
- [Chroma](https://www.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [PyPDF](https://pypdf.readthedocs.io/)
- [docx2txt](https://github.com/ankushshah89/python-docx2txt)
- [html2text](https://github.com/Alir3z4/html2text)
- [ActiveLoop](https://www.activeloop.ai/)
- [Neo4j](https://neo4j.com/)
- [Tavily](https://www.tavily.com/)
- [Exa](https://exa.ai/)
- [Browserbase](https://browserbase.com/)
- [Firecrawl](https://www.firecrawl.dev/)
- [BBB](https://www.bbb.org/)
- [Glean](https://glean.com/)
- [Modus Framework](https://github.com/hypermodeinc/modus)
- [Pipedream](https://pipedream.com/)
- [n8n](https://n8n.io/)
- [Composio](https://composio.dev/)
- [Sim](https://www.sim.ai/)
- [Apify](https://apify.com/)
- [Stagehand](https://www.stagehand.dev/)
- [Browser Use](https://browser-use.com/)
- [Mastra](https://mastra.ai/)
- [Vapi](https://vapi.ai/)
- [Basic](https://basic.tech/)
- [spaCy](https://spacy.io/)
- [Hugging Face](https://huggingface.co/)
- [Stanza](https://stanfordnlp.github.io/stanza/)
- [Flair](https://github.com/flairNLP/flair)
- [REBEL](https://github.com/Babelscape/rebel)
- [KAIROS](https://github.com/isi-nlp/Kairos)
- [DyGIE++](https://github.com/luanyi/DyGIE)
- [Guardrails.ai](https://www.guardrailsai.com/)
- [Rebuff](https://rebuff.ai/)
- [Llama Guard](https://github.com/meta-llama/PurpleLlama)
- [RAGAS](https://github.com/explodinggradients/ragas)
- [Promptlayer](https://promptlayer.com/)
- [Llama Prompt Guard 2](https://www.llama.com/docs/model-cards-and-prompt-formats/prompt-guard/)
- [HarmBench](https://github.com/centerforaisafety/HarmBench)

---

*Organized by use case for easy discovery and implementation. Each tool includes practical applications and integration possibilities.*
