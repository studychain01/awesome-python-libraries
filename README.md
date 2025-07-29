# Awesome Python Libraries & Tools

A curated collection of powerful Python libraries and tools organized by use case, perfect for developers, data scientists, and AI enthusiasts.

## 🤖 AI & Machine Learning Frameworks

### LangChain & RAG (Retrieval-Augmented Generation)
- **LangChain** - Framework for building LLM applications with chunking strategies
  - **Header-based chunking** - Break documents by headers for better context
  - **Sentence-based chunking** - Break documents by sentences for granular processing
  - **Use case**: Document processing, content summarization, Q&A systems

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

### For Business Applications
1. Use **BBB API** for business verification
2. Research companies with **Glean**
3. Build custom agents with **Modus Framework**
4. Extract business entities with **spaCy** or **Flair**

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
- [ActiveLoop](https://www.activeloop.ai/)
- [Neo4j](https://neo4j.com/)
- [Tavily](https://www.tavily.com/)
- [Exa](https://exa.ai/)
- [Firecrawl](https://www.firecrawl.dev/)
- [BBB](https://www.bbb.org/)
- [Glean](https://glean.com/)
- [Modus Framework](https://github.com/hypermodeinc/modus)
- [Pipedream](https://pipedream.com/)
- [n8n](https://n8n.io/)
- [spaCy](https://spacy.io/)
- [Hugging Face](https://huggingface.co/)
- [Stanza](https://stanfordnlp.github.io/stanza/)
- [Flair](https://github.com/flairNLP/flair)
- [REBEL](https://github.com/Babelscape/rebel)
- [KAIROS](https://github.com/isi-nlp/Kairos)
- [DyGIE++](https://github.com/luanyi/DyGIE)

---

*Organized by use case for easy discovery and implementation. Each tool includes practical applications and integration possibilities.*
