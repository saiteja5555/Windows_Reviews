#!/usr/bin/env python
# coding: utf-8

# In[35]:


import streamlit as st
from azure.core.credentials import AzureKeyCredential
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.llms import AzureOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
import openai
import pyodbc
import urllib
from sqlalchemy import create_engine
import pandas as pd
from azure.identity import InteractiveBrowserCredential
from pandasai import SmartDataframe
import pandas as pd
from pandasai.llm import AzureOpenAI
import matplotlib.pyplot as plt
import os
import time
from PIL import Image
import base64
import pandasql as ps
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#Initializing API Keys to use LLM
os.environ["AZURE_OPENAI_API_KEY"] = "b71d4af1ea184bfb9444b448f4f5412a"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://fordmustang.openai.azure.com/"


#Reading the dataset
Sentiment_Data  = pd.read_csv("Windows_Data_116K.csv")

#Function to derive Sentiment Score based on Sentiment
def Sentiment_Score_Derivation(value):
    try:
        if value.lower() == "positive":
            return 1
        elif value.lower() == "negative":
            return -1
        else:
            return 0
    except Exception as e:
        err = f"An error occurred while deriving Sentiment Score: {e}"
        return err    

#Deriving Sentiment Score and Review Count columns into the dataset
Sentiment_Data["Sentiment_Score"] = Sentiment_Data["Sentiment"].apply(Sentiment_Score_Derivation)
Sentiment_Data["Review_Count"] = 1.0


################################# Definiting Functions #################################

#Review Summarization (Detailed) + Feature Comparison and Suggestion

#Function to extract text from file
def get_text_from_file(txt_file):
    try:
        with open(txt_file, 'r',encoding='latin') as file:
            text = file.read()
        return text
    except Exception as e:
        err = f"An error occurred while getting text from file: {e}"
        return err

# Function to split text into chunks
def get_text_chunks(text):
    try:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        err = f"An error occurred while getting text chunks: {e}"
        return err

# Function to create and store embeddings
def get_vector_store(text_chunks):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index_Windows_116k")
        return vector_store
    except Exception as e:
        err = f"An error occurred while getting vectos: {e}"
        return err

# Function to setup the vector store (to be run once or upon text update)
def setup(txt_file_path):
    try:
        raw_text = get_text_from_file(txt_file_path)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        print("Setup completed. Vector store is ready for queries.")
    except Exception as e:
        err = f"An error occurred while setting up vector store: {e}"
        return err

# Function to get conversational chain
def get_conversational_chain_detailed(history):
    try:
        hist = """"""
        for i in history:
            hist = hist+"\nUser: "+i[0]
            if isinstance(i[1],pd.DataFrame):
                x = i[1].to_string()
            else:
                x = i[1]
            hist = hist+"\nResponse: "+x
        prompt_template = """
            Given a dataset with these columns: Review, Data_Source, Geography, Product_Family, Sentiment, and Aspect (also called Features)

            Review: This column contains the opinions and experiences of users regarding different product families across geographies, providing insights into customer satisfaction or complaints and areas for improvement.
            Data_Source: This column indicates the platform from which the user reviews were collected.
            Geography: This column lists the countries of the users who provided the reviews, allowing for an analysis of regional preferences and perceptions of the products.
            Product_Family: This column identifies the broader category of products to which the review pertains, enabling comparisons and trend analysis across different product families.
            Sentiment: This column reflects the overall tone of the review, whether positive, negative, or neutral, and is crucial for gauging customer sentiment.
            Aspect: This column highlights the particular features or attributes of the product that the review discusses, pinpointing areas of strength or concern.
            
            Perform the required task from the list below, as per user's query:

            Review Summarization - Summarize the reviews by filtering the relevant Aspect, Geography, Product_Family, Sentiment, or Data_Source, only based on available reviews and their sentiments in the dataset.
            Aspect Comparison - Provide a summarized comparison for each overlapping feature/aspect between the product families or geographies, only based on available user reviews and their sentiments in the dataset. Include pointers for each aspect highlighting the key differences between the product families or geographies, along with the positive and negative sentiments as per customer perception.
            New Feature Suggestion/Recommendation - Generate feature suggestions or improvements or recommendations based on the frequency and sentiment of reviews and mentioned aspects and keywords. Show detailed responses to user queries by analyzing review sentiment, specific aspects, and keywords.
            Hypothetical Reviews - Based on varying customer sentiments for the reviews in the existing dataset, generate hypothetical reviews for any existing feature updation or new feature addition in any device family across any geography by simulating user reactions. Ensure to synthesize realistic reviews that capture all types of sentiments and opinions of users by considering their hypothetical prior experience working with the new feature and generate output based on data present in dataset only. After these, provide solutions/remedies for negative hypothetical reviews.
            
            Enhance the model’s comprehension to accurately interpret user queries by:

            Recognizing abbreviations for country names (e.g., ‘DE’ for Germany, ‘USA’ or 'usa' or 'US' for the United States of America) and expanding them to their full names for clarity.
            Understanding product family names even when written in reverse order or missing connecting words (e.g., ‘windows in windows 11’ as ‘windows windows’ and ‘windows for security’ as ‘windows security’ etc.).
            Utilizing context and available data columns to infer the correct meaning and respond appropriately to user queries involving variations in product family names or geographical references.
            Please provide a comprehensive Review summary, feature comparison, feature suggestions for specific product families, and actionable insights that can help in product development and marketing strategies.
            Generate accurate response only; do not provide extra information.


            
            Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.\n Following is the previous conversation from User and Response, use it to get context only:""" + hist + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Thruxton_R",
            api_version='2024-03-01-preview',
            temperature = 0.4)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for detailed review summarization: {e}"
        return err

# Function to handle user queries using the existing vector store
def query_detailed(user_question, history, vector_store_path="faiss_index_Windows_116k"):
    try:
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        chain = get_conversational_chain_detailed(history)
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        err = f"An error occurred while getting LLM response for detailed review summarization: {e}"
        return err


## Review Summarization (Quantifiable)

#Converting Top Operator to Limit Operator as pandasql doesn't support Top
def convert_top_to_limit(sql):
    try:
        tokens = sql.upper().split()
        is_top_used = False

        for i, token in enumerate(tokens):
            if token == 'TOP':
                is_top_used = True
                if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                    limit_value = tokens[i + 1]
                    # Remove TOP and insert LIMIT and value at the end
                    del tokens[i:i + 2]
                    tokens.insert(len(tokens), 'LIMIT')
                    tokens.insert(len(tokens), limit_value)
                    break  # Exit loop after successful conversion
                else:
                    raise ValueError("TOP operator should be followed by a number")

        return ' '.join(tokens) if is_top_used else sql
    except Exception as e:
        err = f"An error occurred while converting Top to Limit in SQL Query: {e}"
        return err

#Function to add Table Name into the SQL Query as it is, as the Table Name is Case Sensitive here
def process_tablename(sql, table_name):
    try:
        x = sql.upper()
        query = x.replace(table_name.upper(), table_name)
        return query
    except Exception as e:
        err = f"An error occurred while processing table name in SQL query: {e}"
        return err

## Generating Response by Identifying Prompt Nature

#Function to get conversation chain for quantitative outputs and also add context from historical conversation as well
def get_conversational_chain_quant(history):
    try:
        hist = """"""
        for i in history:
            hist = hist+"\nUser: "+i[0]
            if isinstance(i[1],pd.DataFrame):
                x = i[1].to_string()
            else:
                x = i[1]
            hist = hist+"\nResponse: "+x
        prompt_template = """
        1. Your Job is to convert the user question to SQL Query (Follow Microsoft SQL server SSMS syntax.). You have to give the query so that it can be used on Microsoft SQL server SSMS.You have to only return query as a result.
            2. There is only one table with table name Sentiment_Data where each row is a user review. The table has 10 columns, they are:
                Review: Review of the Copilot Product
                Data_Source: From where is the review taken. 
                Geography: From which Country or Region the review was given.
                Title: What is the title of the review
                Review_Date: The date on which the review was posted
                Product: Corresponding product for the review.
                Product_Family: Which version or type of the corresponding Product was the review posted for.
                Sentiment: What is the sentiment of the review. It contains following values: 'positive', 'neutral', 'negative'.
                Aspect: The review is talking about which aspect or feature of the product.
                Keyword: What are the keywords mentioned in the product.
                Review_Count - It will be 1 for each review or each row.
                Sentiment_Score - It will be 1, 0 or -1 based on the Sentiment.
            3. Sentiment mark is calculated by sum of Sentiment_Score.
            4. Net sentiment is calculcated by sum of Sentiment_Score divided by sum of Review_Count. It should be in percentage. Example:
                    SELECT ((SUM(Sentiment_Score)*1.0)/(SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment 
                    FROM Sentiment_Data
                    ORDER BY Net_Sentiment DESC
            5. Net sentiment across country or across region is sentiment mark of a country divided by total reviews of that country. It should be in percentage.
                Example to calculate net sentiment across country:
                    SELECT Geography, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Sentiment_Data
                    GROUP BY Geography
                    ORDER BY Net_Sentiment DESC
            6. Net Sentiment across a column "X" is calculcated by Sentiment Mark for each "X" divided by Total Reviews for each "X".
                Example to calculate net sentiment across a column "X":
                    SELECT X, ((SUM(Sentiment_Score)*1.0) / (SUM(Review_Count)*1.0)) * 100 AS Net_Sentiment
                    FROM Sentiment_Data
                    GROUP BY X
                    ORDER BY Net_Sentiment DESC
            7. Distribution of sentiment is calculated by sum of Review_Count for each Sentiment divided by overall sum of Review_Count
                Example: 
                    SELECT Sentiment, SUM(ReviewCount)*100/(SELECT SUM(Review_Count) AS Reviews FROM Sentiment_Data) AS Total_Reviews 
                    FROM Sentiment_Data 
                    GROUP BY Sentiment
                    ORDER BY Total_Reviews DESC
            8. Convert numerical outputs to float upto 1 decimal point.
            9. Always include ORDER BY clause to sort the table based on the aggregate value calculated in the query.
            10. Top Country is based on Sentiment_Score i.e., the Country which have highest sum(Sentiment_Score)
            11. Always use 'LIKE' operator whenever they mention about any Country. Use 'LIMIT' operator instead of TOP operator.Do not use TOP OPERATOR. Follow syntax that can be used with pandasql.
            12. If you are using any field in the aggregate function in select statement, make sure you add them in GROUP BY Clause.
            13. Make sure to Give the result as the query so that it can be used on Microsoft SQL server SSMS.
            14. Important: Always show Net_Sentiment in Percentage upto 1 decimal point. Hence always make use of ROUND function while giving out Net Sentiment and Add % Symbol after it.
            15. Important: User can ask question about any categories including Aspects, Geograpgy, Sentiment etc etc. Hence, include the in SQL Query if someone ask it.
            16. Important: You Response should directly starts from SQL query nothing else.
            17. Important: Always use LIKE keyword instead of = symbol while generating SQL query.
            18. Important: Generate outputs using the provided dataset only, don't use pre-trained information to generate outputs.
        \n Following is the previous conversation from User and Response, use it to get context only:""" + hist + """\n
                Use the above conversation chain to gain context if the current prompt requires context from previous conversation.\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Thruxton_R",
            api_version='2024-03-01-preview',
            temperature = 0.3)
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        err = f"An error occurred while getting conversation chain for quantifiable review summarization: {e}"
        return err

#Function to convert user prompt to quantitative outputs for Copilot Review Summarization
def query_quant(user_question, history, vector_store_path="faiss_index_Windows_116k"):
    try:
        # Initialize the embeddings model
        embeddings = AzureOpenAIEmbeddings(azure_deployment="MV_Agusta")
        
        # Load the vector store with the embeddings model
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
        
        # Rest of the function remains unchanged
        chain = get_conversational_chain_quant(history)
        docs = []
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        SQL_Query = response["output_text"]
        SQL_Query = convert_top_to_limit(SQL_Query)
        SQL_Query = process_tablename(SQL_Query,"Sentiment_Data")
#         print(SQL_Query)
        data = ps.sqldf(SQL_Query, globals())
        data_1 = data
        html_table = data.to_html(index=False)
    #     return html_table
        return data_1
    except Exception as e:
        err = f"An error occurred while generating response for quantitative review summarization: {e}"
        return err


## Generating Response by Identifying Prompt Nature


#Function to identify the nature of prompt, whether the user is asking for a detailed summary or a quantitative summary
def identify_prompt(user_question):
    try:
        prompt_template = """
        Given a user prompt about customer reviews for products (Windows Devices) and various different features, classify the prompt into one of two categories:
            Quantifiable: This prompt seeks a numerical answer or data point related to the reviews. 
                            (e.g., "What is the net sentiment score for Product A reviews?", 
                                    "How many reviews mention the battery life of Product B?", 
                                    "Calculate the net sentiment of Product A.", 
                                    "Net Sentiment", 
                                    "Sentiment Score", 
                                    "Top Countries", 
                                    "Top Products", etc.)
            Detailed: This prompt seeks a summary, comparison, recommendation/suggestion or hypothetical reviews based on the reviews, expressed in words. The task can be either Review Summarization, Feature Comparison, New Feature Suggestion/Recommendation or Hypothetical Reviews generation based on the type of user question:
                      (example of Review Summarization - Summarize / Give a summary of the reviews for different product families or geographies, 
                      example of Aspect Comparison - Give feature comparison among product families a1,a2,a3... across geographies g1,g2,g3... or Compare the features of product families a1,a2,a3... across geographies g1,g2,g3,... or Compare utility of feature 'x' among product families a1,a2,a3... across geographies g1,g2,g3... , 
                      example of New Feature Suggestion/Recommendation - Suggest new features or improvements or recommendations for different product families in different geographies,
                      example of Hypothetical Reviews - Generate hypothetical user reviews for the feature upgrade for any product family in any geography, focusing on the feature/aspect or Provide hypothetical user reviews for the addition of the new feature 'x' in any product family across any geography)

        Input: User prompt about customer reviews
        Output: Category (Quantifiable or Detailed)
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        model = AzureChatOpenAI(
            azure_deployment="Thruxton_R",
            api_version='2024-03-01-preview')
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        response = chain({"input_documents": [], "question": user_question}, return_only_outputs=True)
        if "detailed" in response["output_text"].lower():
            return "Detailed"
        elif "quantifiable" in response["output_text"].lower():
            return "Quantifiable"
        else:
            return "Others"+"\nPrompt Identified as:"+response["output_text"]+"\n"
    except Exception as e:
        err = f"An error occurred while generating conversation chain for identifying nature of prompt: {e}"
        return err

#Function to generate Review Summarization (Detailed)/Feature Comparison/Feature Suggestion from User Prompt
def review_summarization(user_question, history):
    try:
        txt_file_path = "Sample_Copilot_Reviews_50K_Translated_v2.txt"
        # Automatically call setup with the predefined file on startup
        if not os.path.exists("faiss_index_Windows_116k"):
            setup(txt_file_path)

        if os.path.exists("faiss_index_Windows_116k"):
            response = query_detailed(user_question, history)
            return response
        else:
            return "The vector store setup has failed. Please check the file path and try again."
    except Exception as e:
        err = f"An error occurred while generating detailed review summarization: {e}"
        return err

#Function to generate Quantitative Review Summarization from User Prompt
def quantifiable_data(user_question, history):
    try:
        response = query_quant(user_question, history)
        
        return response
    except Exception as e:
        err = f"An error occurred while generating quantitative review summarization: {e}"
        return err

#Function to generate a response from User Question
def device_llm_review_generator(user_question, history):
    try:
        identity_prompt = identify_prompt(user_question)
        if identity_prompt == "Detailed":
            output = review_summarization(user_question, history)
        elif identity_prompt == "Quantifiable":
            output = quantifiable_data(user_question, history)

            if output.isnull().all().all() or output.empty:
                output = "I don't have necessary knowledge to provide a helpful response. Please ask anything related to Insights around Windows customer feedback and I'll do my best to assist you."
                
        else:
            output = "Error: Cannot identify the nature of your question\nPrompt identified as: "+identity_prompt
        return output
    except Exception as e:
        err = f"An error occurred while generating LLM response: {e}"
        return err


################################# Model Deployment #################################

def main():
    try:
    # Chat history state management
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []

        # Create a container for logos and title with horizontal layout
        col1, col2, col3 = st.columns([1, 2, 1])
      
        # Display logo on the left
        with col1:
            st.image("microsoft_logo.png", width=50)  # Adjust width as needed

        # Display title in the center
        with col2:
            st.header("Windows LLM Reviews Summarizer")

        # Display logo on the right
        with col3:
            st.image("copilot_logo.svg", width=50)  # Align the logo to the right
      
        # User input section
        user_input = st.text_input("Enter your text:", placeholder="What would you like to process?")

        # Process button and output section
        if st.button("Process"):
            # Re-check if 'chat_history' is initialized before appending
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
            
            output = device_llm_review_generator(user_input,st.session_state['chat_history'])
            st.session_state['chat_history'].append((user_input, output))
        
            # Display output based on type (string or dataframe)
            if isinstance(output, pd.DataFrame):
                st.dataframe(output)
                
            else:
                st.write(output)

        # Chat history section with some formatting
        st.header("Chat History")
        for user_text, output_text in st.session_state['chat_history']:
            st.markdown(f"- You: {user_text}")
            if isinstance(output_text, pd.DataFrame):
                st.dataframe(output_text)  # Convert dataframe to string for display
            else:
                st.markdown(f"- Bot: {output_text}")
            st.write("---")
    except Exception as e:
        err = f"An error occurred while calling the final function: {e}"
        print(err)
        return err


if __name__ == "__main__":
    main()


# In[ ]:




