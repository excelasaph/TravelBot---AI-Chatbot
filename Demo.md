# TravelBot - Video Demo Script

## 1. Introduction (0:45)
- **Project Overview**: "Hello and welcome to my TravelBot demo! I'm Excel Asaph, and today I'll be demonstrating my domain-specific chatbot designed to answer travel and geography questions. This project is built using a fine-tuned T5 Transformer model from Hugging Face, specialized in the travel and geography domain."
- **Purpose**: "The goal was to create an AI assistant that can provide accurate, relevant responses to travel-related queries - from destination information to cultural insights and travel planning advice."
- **Domain Selection**: "I chose the travel and geography domain because it's information-rich, has clear boundaries for knowledge scope, and offers practical value for users planning trips or learning about different places."

## 2. Project Architecture (1:30)
- **Show a diagram or code structure**: "Let me walk you through the project architecture. At its core is a fine-tuned T5 transformer model, which I chose for its text-to-text approach that works well for question-answering tasks."
- **Components**: "The project consists of several key components:
  - A Python backend using FastAPI for model inference
  - A user-friendly Streamlit frontend for chatbot interaction 
  - The fine-tuned T5 model stored in the 'fine_tuned_t5_travel_geography' folder
  - Evaluation metrics and sample predictions in the 'metrics' folder"
- **Show GitHub structure**: "As you can see from my GitHub repository, I've organized the code into logical components with appropriate documentation."

## 3. Dataset Processing (1:30)
- **Dataset selection**: "For training data, I used the BAAI/IndustryInstruction_Travel-Geography dataset from Hugging Face, which contains high-quality travel and geography Q&A pairs."
- **Show notebook data section**: "In my Jupyter notebook, I'll show you how I processed the data:
  - First, I filtered for English conversations from the multilingual dataset
  - I sampled 20,000 entries for quality and computational efficiency
  - Then I applied preprocessing including normalization, lemmatization, and format preparation
  - Finally, I extracted clean query-response pairs for fine-tuning"
- **Show word clouds**: "These word clouds visualize the most common terms in queries and responses, highlighting the travel-focused vocabulary my model learned."

## 4. Model Training and Fine-tuning (1:45)
- **Base model selection**: "I chose the T5-base model as my foundation because:
  - It's designed for text-to-text tasks, making it ideal for Q&A
  - It has strong pre-trained knowledge about geography and locations
  - It strikes a good balance between performance and size for deployment"
- **Show fine-tuning code**: "In the notebook, I'll demonstrate the fine-tuning process:
  - I used Hugging Face's Trainer with PyTorch
  - The model was trained for 5 epochs with batch size 8 and learning rate 5e-5
  - I implemented early stopping to prevent overfitting"
- **Show hyperparameter experiments**: "I experimented with different hyperparameters to optimize performance. As you can see in my experiment tables, I found the best results with these settings."
- **Show training loss curve**: "This training curve shows how the model converged nicely without significant overfitting."

## 5. Model Evaluation (1:30)
- **Metrics overview**: "To evaluate my model, I used multiple complementary metrics:
  - ROUGE scores for measuring text overlap with references
  - BERTScore for semantic similarity assessment
  - BLEU for precision of n-gram matches"
- **Show metrics results**: "Let me share the key results from evaluation_metrics.json:
  - ROUGE-L score of 0.296, showing good coverage of reference content
  - BERTScore F1 of 0.877, indicating strong semantic alignment
  - BLEU score of 0.109, which is reasonable for open-ended generation"
- **Sample predictions**: "Looking at some sample predictions in the metrics folder, we can see how the model performs on real queries."
- **Out-of-domain handling**: "I also implemented an out-of-domain detection system that recognizes when questions are outside the travel domain."

## 6. Application Interface (2:00)
- **Streamlit UI Demo**: "Now let's look at the user interface I built with Streamlit:
  - The chatbot has a clean, intuitive chat interface
  - Users can type travel questions and receive AI-generated responses
  - The system maintains chat history and allows response downloads
  - I've included sample prompts to help users get started"
- **Live demo with examples**: "Let me demonstrate with some example questions:
  1. 'What are the best times to visit Japan?'
  2. 'What should I know about traveling to Machu Picchu?'
  3. 'What are some cultural customs I should be aware of in Thailand?'"
- **Out-of-domain example**: "If I ask something unrelated like 'How do I solve this math equation?', the system recognizes it's outside the travel domain and responds appropriately."
- **Deployment details**: "I deployed the application using Render's cloud platform, making it accessible to users worldwide through a web interface."

## 7. Technical Challenges and Solutions (1:00)
- **Challenge 1**: "One key challenge was tokenizer warnings disrupting the Streamlit connection. I solved this by adding the 'legacy=True' parameter to tokenizer initialization."
- **Challenge 2**: "Balancing model size with performance was tricky. I chose T5-base with optimized generation parameters over larger variants to stay within deployment constraints."
- **Challenge 3**: "Handling out-of-domain queries required developing a custom heuristic based on response length that proved surprisingly effective."
- **Challenge 4**: "Deployment resource constraints on free tier services were overcome by implementing lazy model initialization and proper gunicorn configuration."

## 8. Conclusion and Future Improvements (0:30)
- **Summary**: "In summary, I've successfully built and deployed a domain-specific chatbot for travel and geography using a fine-tuned T5 model."
- **Metrics recap**: "The model achieves strong performance on standard NLP metrics and provides helpful, accurate travel information."
- **Future work**: "Given more time, I would:
  - Implement a more sophisticated out-of-domain detection system
  - Add multilingual support for international travelers
  - Integrate real-time travel data like weather and COVID restrictions
  - Expand the training dataset with more specialized travel content"
- **Thank you**: "Thank you for watching this demonstration of my TravelBot project. The complete code and documentation are available in my GitHub repository."

## Demo Preparation Notes

### Environment Setup
1. Ensure all dependencies are installed: `pip install -r requirements.txt`
2. Verify the fine-tuned model is properly loaded in the application folder
3. Test the Streamlit app locally before recording: `streamlit run app/streamlit_app.py`

### Recording Tips
1. Use screen recording software that captures both screen and audio
2. Prepare example travel questions that showcase different capabilities
3. Have the GitHub repository open in a browser tab for easy reference
4. Ensure the notebook is pre-executed to avoid waiting for cell execution
5. Consider recording sections separately and editing them together
6. Speak clearly and at a moderate pace

### Visual Aids to Include
- Project architecture diagram
- Word clouds from data analysis
- Training loss curves
- Evaluation metrics table
- Sample predictions
- GitHub repository structure
- Streamlit application interface

### Demo Flow Checklist
- [ ] Introduction and project overview
- [ ] Code structure and architecture explanation
- [ ] Dataset description and preprocessing steps
- [ ] Model selection rationale and fine-tuning process
- [ ] Evaluation metrics and result analysis
- [ ] Live demonstration of chatbot interface
- [ ] Technical challenges and solutions
- [ ] Conclusion and future improvements

### Sample Questions for Demo
1. "What are the top tourist attractions in Barcelona?"
2. "How do I prepare for hiking in high altitudes like the Andes?"
3. "What's the best time of year to visit Thailand?"
4. "Can you suggest a 3-day itinerary for Rome?"
5. "What are the visa requirements for visiting Japan as a US citizen?"
6. "What cultural practices should I be aware of when visiting Morocco?"
7. "How can I travel sustainably in Southeast Asia?"
8. "What are the must-try foods in Italy?"
9. OUT-OF-DOMAIN: "Who won the 2024 Olympics?"
10. OUT-OF-DOMAIN: "How do I fix my computer?"