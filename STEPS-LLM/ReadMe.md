## Knowledge Hub LLM System

This project is part of a larger initiative by the MS department to build a global **Knowledge Hub** that can be accessed by external researchers worldwide. The system is designed in multiple stages:

1. **Synthetic Data & Dummy Reports**  
   - Generate synthetic datasets and dummy research reports using the **OpenAI API**.  
   - Fine-tune LLMs so they can extract information as well as learn relationships between inputs, outputs, and experimental parameters.  
   - In later phases, replace synthetic data with **real research papers** to enhance domain-specific understanding.  

2. **Multi-Agent Monitoring & Guardrails**  
   - Develop a **monitoring and scoring system** that evaluates LLM outputs.  
   - Use multiple agents to impose guardrails, preventing nonsensical or hallucinated answers.  

### Current Progress
- Implemented **synthetic report generation** using OpenAI API keys.  
- Built a **Retrieval-Augmented Generation (RAG) system** with memory to maintain context across conversations.  
- Created a **Streamlit interface** to interact with the RAG system.  
- Future work includes fine-tuning LLMs on research papers so they can reason beyond the training context.  

### Skills Demonstrated
- Synthetic data generation & augmentation  
- Retrieval-Augmented Generation with persistent memory  
- Fine-tuning LLMs for domain-specific tasks  
- Multi-agent system design for monitoring and guardrails  
- Applied use of **OpenAI API** and **Streamlit**
