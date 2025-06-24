
"""
Jupiter FAQ Bot Application

This application scrapes FAQs from Jupiter Money website, preprocesses the data,
and provides a conversational AI interface to answer user questions.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
    "Help": "/help",
    "Contact us": "/contact-us"
}


def scrape_all_faqs_corrected(url_map):
    """
    Scrapes FAQs from a dictionary of Jupiter pages, handling multiple HTML structures.
    """
    base_url = "https://jupiter.money"
    all_faq_data = []

    for category, path in url_map.items():
        url = f"{base_url}{path.strip()}"
        print(f"Scraping category: '{category}' from {url}")

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"  -> Failed to fetch {url}: {e}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the FAQ section heading (case-insensitive search for flexibility)
        faq_heading = soup.find(['h1', 'h2'], string=lambda text: text and 'frequently asked questions' in text.lower())

        if not faq_heading:
            print(f"  -> No FAQ heading found.")
            continue

        # --- Adaptive Scraping Logic ---
        extracted_count = 0

        # Strategy 1: Look for the new structure (e.g., on /bills-recharges)
        # The container is often the next sibling div of the heading.
        container = faq_heading.find_next_sibling('div')
        if container:
            # The new structure uses 'faq-item' for each Q&A
            qa_items = container.find_all('div', class_='faq-item')
            if qa_items:
                print(f"  -> Found new structure with {len(qa_items)} items.")
                for item in qa_items:
                    # Question is in a span inside the 'faq-header'
                    question_tag = item.find('div', class_='faq-header').find('span')
                    # Answer is in a div with class 'faq-answer'
                    answer_div = item.find('div', class_='faq-answer')

                    if question_tag and answer_div:
                        question = question_tag.get_text(strip=True)
                        answer = answer_div.get_text(strip=True, separator='\n')
                        all_faq_data.append({"category": category, "question": question, "answer": answer})
                        extracted_count += 1

        # Strategy 2: Fallback to the old structure (e.g., on /help) if Strategy 1 found nothing
        if extracted_count == 0:
            # The old structure used a different set of classes
            qa_items_fallback = soup.find_all('div', class_='jupiter-help-center-accordion-item-content-qa')
            if qa_items_fallback:
                print(f"  -> Found old structure with {len(qa_items_fallback)} items.")
                for qa in qa_items_fallback:
                    question_tag = qa.find('p', class_='jupiter-help-center-accordion-item-content-qa-title')
                    answer_div = qa.find('div', class_='jupiter-help-center-accordion-item-content-qa-desc')

                    if question_tag and answer_div:
                        question = question_tag.get_text(strip=True)
                        answer = answer_div.get_text(strip=True, separator='\n')
                        all_faq_data.append({"category": category, "question": question, "answer": answer})
                        extracted_count += 1
        
        if extracted_count > 0:
            print(f"  -> Successfully extracted {extracted_count} Q&A pairs.")
        else:
            print(f"  -> Found FAQ heading, but could not extract Q&A pairs with known structures.")

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
    def __init__(self, dataframe, model_name='all-MiniLM-L6-v2'):
        if dataframe.empty:
            raise ValueError("The provided DataFrame is empty. Cannot initialize the bot.")
        self.df = dataframe
        self.model = SentenceTransformer(model_name)
        print("Encoding questions into embeddings...")
        self.question_embeddings = self.model.encode(self.df['cleaned_question'].tolist())
        print("Embeddings created.")

        # Build FAISS index
        self.index = faiss.IndexFlatL2(self.question_embeddings.shape[1])
        self.index.add(np.array(self.question_embeddings, dtype=np.float32))
        print("FAISS index created.")

        # Initialize the LLM
        self.llm = genai.GenerativeModel('gemini-2.5-pro')

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
            # Note: Ensure your API key is configured before this step.
            response = self.llm.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating response from LLM: {e}")
            # Fallback to the direct answer if LLM fails
            return f"I found this information which might help:\n\n{retrieved_answer}"


def chat_with_bot(faq_bot):
    """
    A simple command-line interface to chat with the FAQ bot.
    """
    if not faq_bot:
        print("Bot could not be initialized. Exiting.")
        return

    print("\n--- Jupiter FAQ Bot ---")
    print("Ask me anything about Jupiter's services. Type 'exit' to quit.")
    print("-" * 25)

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Bot: Happy to help. Goodbye!")
            break
        bot_response = faq_bot.get_conversational_answer(user_input)
        print(f"Bot: {bot_response}\n")


def main():
    """
    Main function to execute the complete FAQ bot workflow.
    """
    print("--- Starting Comprehensive FAQ Scraping ---")
    
    # Step 1: Scrape FAQs
    faq_df_corrected = scrape_all_faqs_corrected(URL_MAPPING)

    if not faq_df_corrected.empty:
        print("\n--- Scraping Complete ---")
        print(f"Total FAQs scraped: {len(faq_df_corrected)}")
        print("\nSample of scraped data:")
        print(faq_df_corrected.head())
        # Save the comprehensive scraped data
        faq_df_corrected.to_csv("jupiter_faqs_comprehensive.csv", index=False)
        
        # Step 2: Preprocess the data
        preprocessed_df = preprocess_faqs(faq_df_corrected)
        print("\nPreprocessing complete!")
        print(f"Total FAQs after preprocessing: {len(preprocessed_df)}")
        print(preprocessed_df.head())
        # Save the preprocessed data
        preprocessed_df.to_csv("preprocessed_jupiter_faqs.csv", index=False)
        
        # Step 3: Initialize the FAQ bot
        try:
            faq_bot = FAQBot(preprocessed_df.dropna())
            print("\nFAQ Bot is ready!")
            
            # Step 4: Start the chat interface
            chat_with_bot(faq_bot)
            
        except (ValueError, Exception) as e:
            print(f"\nError initializing bot: {e}")
            
    else:
        print("\n--- Scraping Finished ---")
        print("No data was collected. The website structure may have changed, or the target pages do not contain FAQs in the expected format.")
        
        # Try to load existing preprocessed data
        try:
            preprocessed_df = pd.read_csv("preprocessed_jupiter_faqs.csv").dropna()
            faq_bot = FAQBot(preprocessed_df)
            print("\nLoaded existing FAQ data. Bot is ready!")
            chat_with_bot(faq_bot)
        except FileNotFoundError:
            print("\nNo existing FAQ data found. Please check your internet connection and try again.")


if __name__ == '__main__':
    main()
