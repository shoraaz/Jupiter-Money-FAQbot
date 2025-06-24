#!/usr/bin/env python3
"""
Jupiter FAQ Bot - Streamlit Web Application

This is a Streamlit web interface for the Jupiter FAQ Bot that scrapes FAQs 
from Jupiter Money website and provides conversational AI responses using Gemini API.
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import time

# Set page config
st.set_page_config(
    page_title="Jupiter FAQ Bot",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL mapping for Jupiter Money pages
URL_MAPPING = {
    "Savings account": "/savings-account",
    "Salary account": "/pro-salary-account/",
    "Corporate Salary account": "/corporate-salary-account",
    "Pots": "/pots",
    "Payments": "/payments",
    "Bills & Recharges": "/bills-recharges",
    "Pay via UPI": "/pay-via-upi",
    "Edge+ CSB Bank RuPay credit card": "/edge-plus-upi-rupay-credit-card/",
    "Edge CSB Bank RuPay credit card": "/edge-csb-rupay-credit-card/",
    "Edge Federal Bank VISA credit card": "/edge-visa-credit-card/",
    "Rewards": "/rewards",
    "Loans": "/loan",
    "Loan against mutual funds": "/loan-against-mutual-funds",
    "Investments": "/investments",
    "Mutual Funds": "/mutual-funds",
    "DigiGold": "/digi-gold",
    "Fixed Deposits": "/flexi-fd",
    "Recurring Deposits": "/recurring-deposits",
    "Help": "/contact-us",
    "Contact us": "/contact-us"
}


@st.cache_data
def scrape_all_faqs_corrected(url_map):
    """
    Scrapes FAQs from a dictionary of Jupiter pages, handling multiple HTML structures.
    """
    base_url = "https://jupiter.money"
    all_faq_data = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_categories = len(url_map)
    
    for i, (category, path) in enumerate(url_map.items()):
        url = f"{base_url}{path.strip()}"
        status_text.text(f"Scraping category: '{category}' ({i+1}/{total_categories})")
        progress_bar.progress((i+1) / total_categories)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            st.warning(f"Failed to fetch {category}: {e}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the FAQ section heading (case-insensitive search for flexibility)
        faq_heading = soup.find(['h1', 'h2'], string=lambda text: text and 'frequently asked questions' in text.lower())

        if not faq_heading:
            continue

        # --- Adaptive Scraping Logic ---
        extracted_count = 0

        # Strategy 1: Look for the new structure (e.g., on /bills-recharges)
        container = faq_heading.find_next_sibling('div')
        if container:
            qa_items = container.find_all('div', class_='faq-item')
            if qa_items:
                for item in qa_items:
                    question_tag = item.find('div', class_='faq-header').find('span')
                    answer_div = item.find('div', class_='faq-answer')

                    if question_tag and answer_div:
                        question = question_tag.get_text(strip=True)
                        answer = answer_div.get_text(strip=True, separator='\n')
                        all_faq_data.append({"category": category, "question": question, "answer": answer})
                        extracted_count += 1

        # Strategy 2: Fallback to the old structure
        if extracted_count == 0:
            qa_items_fallback = soup.find_all('div', class_='jupiter-help-center-accordion-item-content-qa')
            if qa_items_fallback:
                for qa in qa_items_fallback:
                    question_tag = qa.find('p', class_='jupiter-help-center-accordion-item-content-qa-title')
                    answer_div = qa.find('div', class_='jupiter-help-center-accordion-item-content-qa-desc')

                    if question_tag and answer_div:
                        question = question_tag.get_text(strip=True)
                        answer = answer_div.get_text(strip=True, separator='\n')
                        all_faq_data.append({"category": category, "question": question, "answer": answer})
                        extracted_count += 1

    progress_bar.empty()
    status_text.empty()
    return pd.DataFrame(all_faq_data)


def clean_text(text):
    """
    Cleans a given text by removing HTML tags and normalizing it.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = " ".join(text.split())
    return text


@st.cache_data
def preprocess_faqs(df):
    """
    Preprocesses the FAQ DataFrame by cleaning and deduplicating.
    """
    if df.empty:
        return df
    
    # Apply the cleaning function to question and answer columns
    df['cleaned_question'] = df['question'].apply(clean_text)
    df['cleaned_answer'] = df['answer'].apply(clean_text)

    # Remove rows where the question or answer is empty after cleaning
    df.dropna(subset=['cleaned_question', 'cleaned_answer'], inplace=True)
    df = df[df['cleaned_question'] != '']

    # Deduplicate based on the cleaned question
    df.drop_duplicates(subset=['cleaned_question'], inplace=True)

    return df


class FAQBot:
    def __init__(self, dataframe, gemini_api_key, model_name='all-MiniLM-L6-v2'):
        if dataframe.empty:
            raise ValueError("The provided DataFrame is empty. Cannot initialize the bot.")
        
        self.df = dataframe
        self.gemini_api_key = gemini_api_key
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.llm = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize sentence transformer
        with st.spinner("Loading sentence transformer model..."):
            self.model = SentenceTransformer(model_name)
        
        # Create embeddings
        with st.spinner("Creating embeddings for FAQ questions..."):
            self.question_embeddings = self.model.encode(self.df['cleaned_question'].tolist())
        
        # Build FAISS index
        with st.spinner("Building FAISS search index..."):
            self.index = faiss.IndexFlatL2(self.question_embeddings.shape[1])
            self.index.add(np.array(self.question_embeddings, dtype=np.float32))

    def find_best_match_index(self, user_query, k=1):
        """Finds the index of the best matching question."""
        query_embedding = self.model.encode([user_query])
        distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), k)
        # Check if the distance is within a reasonable threshold
        if distances[0][0] > 1.5:  # This threshold may need tuning
            return None
        return indices[0][0]

    def get_conversational_answer(self, user_query):
        """Retrieves and rephrases the best answer for a user query."""
        best_match_index = self.find_best_match_index(user_query)

        if best_match_index is None:
            return "I'm sorry, but I couldn't find a specific answer to your question in my knowledge base. Could you try rephrasing it?"

        retrieved_question = self.df.iloc[best_match_index]['question']
        retrieved_answer = self.df.iloc[best_match_index]['answer']

        prompt = f"""
        You are a friendly and helpful assistant for Jupiter, a digital banking app.
        A user has asked the following question: "{user_query}"

        I have found the most relevant FAQ from our knowledge base:
        Original Question: "{retrieved_question}"
        Original Answer: "{retrieved_answer}"

        Your task is to rephrase the "Original Answer" into a simple, friendly, and conversational response.
        - Do NOT just repeat the answer. Make it sound natural and helpful.
        - If the answer lists steps, present them clearly using bullet points or numbered lists.
        - Address the user's query directly.
        - Be confident and clear.
        - If the original answer is very short, you can elaborate slightly to be more helpful, but stay on topic.
        """

        try:
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Error generating response from Gemini: {e}")
            # Fallback to the direct answer if Gemini fails
            return f"I found this information which might help:\n\n{retrieved_answer}"


def main():
    st.title("🏦 Jupiter FAQ Bot")
    st.markdown("### Ask me anything about Jupiter's banking services!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Gemini API Key input
        gemini_api_key = st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="Get your API key from https://makersuite.google.com/app/apikey"
        )
        
        if not gemini_api_key:
            st.warning("Please enter your Gemini API key to continue.")
            st.stop()
        
        st.success("✅ API Key provided!")
        
        # Data management section
        st.header("📊 Data Management")
        
        # Check for existing data
        if os.path.exists("jupiter_faqs_comprehensive.csv") and os.path.exists("preprocessed_jupiter_faqs.csv"):
            st.success("✅ Existing FAQ data found!")
            use_existing = st.radio(
                "Choose data source:",
                ["Use existing data", "Scrape fresh data"],
                help="Using existing data is faster, scraping gives you the latest FAQs"
            )
        else:
            use_existing = "Scrape fresh data"
            st.info("No existing data found. Will scrape fresh data.")
          # Scraping options
        if use_existing == "Scrape fresh data":
            if st.button("🔄 Scrape FAQ Data", type="primary"):
                with st.spinner("Scraping FAQ data from Jupiter Money website..."):
                    try:
                        faq_df = scrape_all_faqs_corrected(URL_MAPPING)
                        if not faq_df.empty:
                            faq_df.to_csv("jupiter_faqs_comprehensive.csv", index=False)
                            
                            preprocessed_df = preprocess_faqs(faq_df)
                            preprocessed_df.to_csv("preprocessed_jupiter_faqs.csv", index=False)
                            
                            st.success(f"✅ Scraped {len(faq_df)} FAQs successfully!")
                            st.rerun()
                        else:
                            st.error("❌ No data was scraped. Please check your internet connection.")
                    except Exception as e:
                        st.error(f"❌ Error during scraping: {e}")

    # Main application logic
    try:
        # Load data
        if use_existing == "Use existing data":
            preprocessed_df = pd.read_csv("preprocessed_jupiter_faqs.csv").dropna()
        else:
            if os.path.exists("preprocessed_jupiter_faqs.csv"):
                preprocessed_df = pd.read_csv("preprocessed_jupiter_faqs.csv").dropna()
            else:
                st.warning("⚠️ Please scrape the data first using the sidebar.")
                st.stop()
        
        # Initialize bot in session state
        if 'faq_bot' not in st.session_state or st.session_state.get('api_key') != gemini_api_key:
            try:
                st.session_state.faq_bot = FAQBot(preprocessed_df, gemini_api_key)
                st.session_state.api_key = gemini_api_key
                st.success("🤖 FAQ Bot initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing bot: {e}")
                st.stop()
        
        # Chat interface
        st.header("💬 Chat with the Bot")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    st.write(answer)
        
        # Chat input
        user_question = st.chat_input("Ask your question about Jupiter's services...")
        
        if user_question:
            # Add user message to chat
            with chat_container:
                with st.chat_message("user"):
                    st.write(user_question)
                
                # Generate bot response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        bot_response = st.session_state.faq_bot.get_conversational_answer(user_question)
                    st.write(bot_response)
              # Add to chat history
            st.session_state.chat_history.append((user_question, bot_response))
            
            # Rerun to update the display
            st.rerun()
        
        # Display data statistics
        with st.expander("📈 FAQ Database Statistics"):
            st.write(f"**Total FAQs in database:** {len(preprocessed_df)}")
            st.write(f"**Categories covered:** {preprocessed_df['category'].nunique()}")
            
            # Category breakdown
            category_counts = preprocessed_df['category'].value_counts()
            st.bar_chart(category_counts)
            
            # Sample questions
            st.subheader("Sample Questions from Database:")
            sample_questions = preprocessed_df['question'].sample(min(5, len(preprocessed_df))).tolist()
            for i, q in enumerate(sample_questions, 1):
                st.write(f"{i}. {q}")
    
    except FileNotFoundError:
        st.error("❌ FAQ data not found. Please scrape the data first using the sidebar.")
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")

    # Footer
    st.markdown("---")
    st.markdown("**Jupiter FAQ Bot** - Powered by Gemini AI and Streamlit")


if __name__ == "__main__":
    main()
