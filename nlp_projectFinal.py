import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import torch

# Page Configuration
st.set_page_config(page_title="Job Recommendation System", page_icon=":briefcase:", layout="wide")

# Caching the data loading and model initialization to improve performance
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_data
def load_job_data():
    # Replace with your actual CSV path
    job_data = pd.read_csv('nlpfinaldataset.csv')
    
    # Text cleaning function
    def clean_text(text):
        if isinstance(text, str):
            text = re.sub(r'[^A-Za-z0-9\s]', '', text)
            text = text.lower()
            text = re.sub(r'\s+', ' ', text).strip()
        else:
            text = ''
        return text
    
    # Clean data
    job_data['cleaned_jobdescription'] = job_data['jobdescription'].apply(clean_text)
    job_data['cleaned_skills'] = job_data['skills'].apply(clean_text)
    job_data['combined_features'] = job_data['jobtitle'] + ' ' + job_data['cleaned_jobdescription'] + ' ' + job_data['cleaned_skills']
    
    return job_data

# Load model and data
model = load_model()
job_data = load_job_data()

# Encode job descriptions once
job_embeddings = model.encode(job_data['combined_features'].tolist(), convert_to_tensor=True)

def recommend_jobs_bert(user_profile, job_data, job_embeddings, location=None, top_k=5):
    # Create user embedding
    user_embedding = model.encode(user_profile, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarities = util.pytorch_cos_sim(user_embedding, job_embeddings)
    
    # Get top K similar job indices
    similar_indices = similarities.argsort(descending=True).tolist()[0][:top_k]
    
    # Get recommended jobs
    recommended_jobs = job_data[['jobtitle', 'joblocation_address']].iloc[similar_indices]
    
    # Optional location filtering
    if location:
        recommended_jobs = recommended_jobs[recommended_jobs['joblocation_address'].str.contains(location, case=False, na=False)]
    
    return recommended_jobs

# Streamlit App
def main():
    st.title("🔍 Job Recommendation System")
    
    # Sidebar for inputs
    st.sidebar.header("Job Search Parameters")
    user_profile = st.sidebar.text_input("Enter your skills/job title", 
                                         placeholder="e.g., Data Scientist with Python and machine learning skills")
    location_filter = st.sidebar.text_input("Optional: Filter by Location", 
                                            placeholder="e.g., New York, San Francisco")
    
    # Recommendation Button
    if st.sidebar.button("Find Recommended Jobs"):
        if user_profile:
            # Get recommendations
            try:
                recommendations = recommend_jobs_bert(
                    user_profile, 
                    job_data, 
                    job_embeddings, 
                    location=location_filter
                )
                
                # Display recommendations
                st.subheader("Recommended Jobs")
                if not recommendations.empty:
                    st.dataframe(recommendations)
                else:
                    st.warning("No jobs found matching your criteria.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter your skills or job title.")
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.info("""
    This Job Recommendation System uses advanced NLP techniques to match 
    your profile with the most relevant job opportunities.
    """)

if __name__ == "__main__":
    main()
