import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

file_path = '/content/nlpfinaldataset.csv'
job_data = pd.read_csv('/content/nlpfinaldataset.csv')

# Modify the clean_text function to handle non-string values
def clean_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    else:
        text = ''  # If it's not a string, return an empty string
    return text

# Apply text cleaning to job titles and descriptions
job_data['cleaned_jobdescription'] = job_data['jobdescription'].apply(clean_text)
job_data['cleaned_skills'] = job_data['skills'].apply(clean_text)

# Create a combined feature from job title, description, and skills
job_data['combined_features'] = job_data['jobtitle'] + ' ' + job_data['cleaned_jobdescription'] + ' ' + job_data['cleaned_skills']
# Display cleaned dataset
job_data[['jobtitle', 'cleaned_jobdescription', 'cleaned_skills']].head()

"""####Model Selection"""

from sklearn.metrics.pairwise import cosine_similarity

# Extract features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(job_data['combined_features'])

# Function to recommend jobs based on the input job title
def recommend_jobs(job_title, job_data, tfidf_matrix):
    idx = job_data[job_data['jobtitle'].str.contains(job_title, case=False)].index[0]

# Example: Get the index of a job title input by the user
job_title_input = 'Data Scientist'  # Replace with the actual user input

# Find the index of the job that matches the user's input
idx = job_data[job_data['jobtitle'].str.contains(job_title_input, case=False)].index[0]

# Calculate cosine similarity between the input job and all other jobs
cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

# Get top 5 similar jobs based on cosine similarity
similar_indices = cosine_sim.argsort()[-6:][::-1]

# Display recommended jobs
recommended_jobs = job_data[['jobtitle', 'joblocation_address']].iloc[similar_indices]
print(recommended_jobs)

"""####Bert for Semantic Similarity"""

!pip install sentence-transformers

from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Example user profile input
user_profile = "Data Scientist with Python and machine learning skills"

# Create the user embedding
user_embedding = model.encode(user_profile, convert_to_tensor=True)

# Encode job descriptions
job_embeddings = model.encode(job_data['combined_features'].tolist(), convert_to_tensor=True)

# Calculate cosine similarity between user profile and job descriptions
similarities = util.pytorch_cos_sim(user_embedding, job_embeddings)

# Function to recommend jobs using BERT embeddings
def recommend_jobs_bert(user_profile, job_data, job_embeddings):
    user_embedding = model.encode(user_profile, convert_to_tensor=True)

# Calculate cosine similarity between user profile and job descriptions
similarities = util.pytorch_cos_sim(user_embedding, job_embeddings)

def some_function():
    # Get top 5 similar jobs
    similar_indices = similarities.argsort(descending=True).tolist()[0][:5]

    # Ensure the return statement is correctly indented within the function
    return job_data[['jobtitle', 'joblocation_address']].iloc[similar_indices]

def recommend_jobs_bert(user_profile, job_data, job_embeddings):
    # Existing logic for job recommendation

    # Ensure you have the logic to get recommended jobs
    recommended_jobs = job_data[['jobtitle', 'joblocation_address']]  # Example

    # Make sure to return the recommended jobs, not None
    return recommended_jobs

print(job_data.head())
print(job_embeddings.shape)

user_profile = "Data Scientist - Houston"
recommended_jobs = recommend_jobs_bert(user_profile, job_data, job_embeddings)

# Print out the recommendations
print(recommended_jobs)

def recommend_jobs(similarities, job_data, location=None):
    # Get top 5 similar jobs
    similar_indices = similarities.argsort(descending=True).tolist()[0][:5]

    # Get recommended jobs
    recommended_jobs = job_data[['jobtitle', 'joblocation_address']].iloc[similar_indices]

    # If a location is provided, filter the recommended jobs by that location
    if location:
        recommended_jobs = recommended_jobs[recommended_jobs['joblocation_address'].str.contains(location, case=False, na=False)]

    return recommended_jobs

# Call the function without location filtering
top_jobs = recommend_jobs(similarities, job_data)

# Call the function with location filtering
top_jobs_in_ny = recommend_jobs(similarities, job_data, location='New York')

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

true_recommendations = [
    "AUTOMATION TEST ENGINEER",  # True top recommendation for the first user/job
    "Information Security Engineer",
    "Business Solutions Architect",
    "Java Developer",
    "DevOps Engineer"
    ]

# Function to calculate Precision@K
def precision_at_k(recommended, true_recommended, k=5):
    recommended_top_k = recommended[:k]
    relevant_count = sum(1 for job in recommended_top_k if job in true_recommended)
    return relevant_count / k

# Function to calculate Recall@K
def recall_at_k(recommended, true_recommended, k=5):
    recommended_top_k = recommended[:k]
    relevant_count = sum(1 for job in recommended_top_k if job in true_recommended)
    return relevant_count / len(true_recommended)

# Function to calculate Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(recommended, true_recommended):
    for idx, job in enumerate(recommended, start=1):
        if job in true_recommended:
            return 1 / idx
            return 0

recommended_jobs = [
    "Java Developer",       # Example recommended jobs, replace with actual recommendations
    "Data Scientist",
    "AUTOMATION TEST ENGINEER",
    "Information Security Engineer",
    "Network Engineer"
]

# Calculate metrics
precision = precision_at_k(recommended_jobs, true_recommendations, k=5)
recall = recall_at_k(recommended_jobs, true_recommendations, k=5)
mrr = mean_reciprocal_rank(recommended_jobs, true_recommendations)

print(f"Precision@K: {precision:.2f}")

print(f"Recall@K: {recall:.2f}")

print(f"Mean Reciprocal Rank (MRR): {mrr:.2f}")

predicted_recommendations = top_jobs[:5]  # Top 5 recommendations

top_jobs = [
    {"jobtitle": "AUTOMATION TEST ENGINEER", "location": "Atlanta, GA"},
    {"jobtitle": "Information Security Engineer", "location": "Chicago, IL"},
]

predicted_recommendations = [job['jobtitle'] for job in top_jobs[:5]]  # Top 5 recommendations

print(top_jobs[:5])

def evaluate_recommendations(true_recommendations, predicted_recommendations):
    # Convert to binary form (1 for correct recommendation, 0 for incorrect)
    y_true = [1 if job in true_recommendations else 0 for job in predicted_recommendations]
    y_pred = [1 if job in predicted_recommendations else 0 for job in predicted_recommendations]

    # Calculate Precision, Recall, F1, and Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1
