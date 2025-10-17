# TravelBot — Travel & Geography Chatbot

<div align="center">
  <img src="./images/TravelBot%20-%20app%201.png" width="700">
</div>

## Project Overview

TravelBot is a domain-specific chatbot fine-tuned for travel and geography question-answering. This project demonstrates the application of transformer models for creating a specialized AI assistant capable of providing accurate information about travel destinations, geographical features, cultural information, and other travel-related topics.

### Demo Video: 
**YouTube Link**: [https://youtu.be/FuIBQ21aULw](https://youtu.be/FuIBQ21aULw)

### Features
- **Domain-Specific Knowledge**: Specialized in travel and geography topics
- **User Interface**: Streamlit-based chat interface for easy interaction
- **Out-of-Domain Detection**: Identifies and gracefully handles off-topic questions
- **API**: FastAPI backend with comprehensive documentation
- **Evaluation**: Performance assessed using ROUGE, BLEU, and BERTScore metrics

## Repository Structure

```
TravelBot---AI-Chatbot/
├── app/                          
│   └── streamlit_app.py          
├── data/                         
│   └── preprocessed_travel_geography_20k.csv  
├── demo/                         
├── fine_tuned_t5_travel_geography/  
├── images/                       
├── metrics/                      
├── report/                       
├── main.py                       
├── render.yaml                   
├── requirements.txt              
├── travel_chatbot.ipynb          
└── README.md                     
```

## Implementation

### Model Architecture
The chatbot is built on a fine-tuned T5 (Text-to-Text Transfer Transformer) model from Hugging Face. After exploring various models, T5 was selected for its:

- **Text-to-Text Framework**: Simplified approach to NLP tasks
- **Pre-training Advantage**: Rich world knowledge beneficial for geography topics
- **Encoder-Decoder Architecture**: Effective for understanding queries and generating responses
- **Size-Performance Balance**: 220M parameters provide good performance while remaining deployable

### Dataset
The model was fine-tuned on the [BAAI/IndustryInstruction_Travel-Geography](https://huggingface.co/datasets/BAAI/IndustryInstruction_Travel-Geography) dataset:

- **Curated Sample**: 20,000 English query-response pairs extracted from the original multilingual dataset
- **Preprocessing**: Includes text normalization, lemmatization, and formatting for T5 input
- **Topic Coverage**: Spans geographical features, travel destinations, cultural information, transportation, and accommodation

### Performance Metrics
The model achieves strong results across multiple metrics:

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.5023 |
| ROUGE-2 | 0.3456 |
| ROUGE-L | 0.4729 |
| BERTScore (F1) | 0.8376 |
| BLEU | 0.2537 |

## Hyperparameter Tuning

Extensive experimentation was conducted to optimize model performance:

**Learning Rate Variations:**

| Learning Rate | Training Loss | Validation Loss | ROUGE-L | BERTScore | BLEU |
|---------------|---------------|----------------|---------|-----------|------|
| 1e-4          | 1.87          | 2.14           | 0.41    | 0.81      | 0.22 |
| 5e-5          | 1.62          | 1.89           | 0.47    | 0.84      | 0.25 |
| 2e-5          | 1.98          | 2.06           | 0.44    | 0.82      | 0.23 |

**Batch Size and Epoch Analysis:**

| Batch Size | Epochs | Training Loss | Validation Loss | ROUGE-L | BERTScore | BLEU |
|------------|--------|---------------|----------------|---------|-----------|------|
| 4          | 3      | 1.58          | 1.92           | 0.45    | 0.83      | 0.24 |
| 8          | 3      | 1.62          | 1.89           | 0.47    | 0.84      | 0.25 |
| 16         | 3      | 1.71          | 1.95           | 0.43    | 0.82      | 0.23 |
| 8          | 5      | 1.40          | 1.85           | 0.48    | 0.85      | 0.26 |

**Final Configuration:** Batch size 8, 5 epochs, learning rate 5e-5, with generation parameters max_length=180, num_beams=4, repetition_penalty=1.8.

## Sample Conversations

**Travel Information:**
> **User:** What is the best time to visit Japan?  
> **TravelBot:** The best time to visit Japan depends on what you want to see and do. Spring (March to May) is popular for cherry blossoms, with peak bloom typically in late March to early April. Fall (September to November) offers beautiful autumn foliage. Summer (June to August) can be hot and humid with occasional typhoons, but it's good for hiking in mountainous regions. Winter (December to February) is cold but ideal for winter sports and seeing snow monkeys. If you want to avoid crowds, consider visiting during the shoulder seasons of late May-June or September-October.

**Geographical Knowledge:**
> **User:** What are the major rivers in South America?  
> **TravelBot:** South America is home to several major river systems. The Amazon River is the largest and most significant, flowing through Peru, Colombia, and Brazil before emptying into the Atlantic Ocean. It's the world's largest river by discharge volume. Other major rivers include the Paraná River (second longest in South America, flowing through Brazil, Paraguay, and Argentina), the Orinoco River (flowing through Venezuela and Colombia), the São Francisco River (in Brazil), and the Magdalena River (Colombia's principal river). The Uruguay River forms part of the borders between Brazil, Argentina, and Uruguay.

**Out-of-Domain Handling:**
> **User:** How do I solve a quadratic equation?  
> **TravelBot:** I can only answer questions about travel and geography. Please ask a question related to that topic.

## Local Setup

1. **Create a virtual environment** and install dependencies:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Run the Streamlit app**:
   ```powershell
   streamlit run app/streamlit_app.py
   ```

3. **Run the FastAPI backend** (optional, for API access):
   ```powershell
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Deployment

The project is deployed using:
- **Streamlit Cloud**: Hosts the user interface at [travelaibot.streamlit.app](https://travelaibot.streamlit.app)
- **Render**: Hosts the FastAPI backend at [travelbot-ai-chatbot.onrender.com](https://travelbot-ai-chatbot.onrender.com)

## Screenshots

Below are screenshots of the TravelBot application interface:

### Main Application Interface
![TravelBot Application Interface](./images/TravelBot%20-%20app%201.png)

### User Interaction Example
![TravelBot User Interaction](./images/TravelBot%20-%20app%202.png)

### Out-of-Domain Check
![TravelBot Advanced Features](./images/TravelBot%20-%20app%203.png)
