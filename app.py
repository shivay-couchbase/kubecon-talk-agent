import streamlit as st
import json
from openai import OpenAI
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
from couchbase.cluster import Cluster
from couchbase.options import ClusterOptions, ClusterTimeoutOptions
from couchbase.auth import PasswordAuthenticator
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.search import SearchRequest, MatchNoneQuery
from datetime import timedelta
import time

# Load environment variables
load_dotenv()

# Initialize OpenAI client for embeddings
# embedding_client = OpenAI(
#     base_url="https://api.studio.nebius.com/v1/",
#     api_key=os.getenv("NEBIUS_API_KEY")
# )

client =  OpenAI(
  base_url="https://api.studio.nebius.com/v1/",
    api_key=""
     
     )

# # Initialize OpenAI client for chat completions
# chat_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class CouchbaseConnection:
    def __init__(self):
        try:
            # Connection details
            connection_string = os.getenv('CB_CONNECTION_STRING')
            username = os.getenv('CB_USERNAME')
            password = os.getenv('CB_PASSWORD')
            bucket_name = os.getenv('CB_BUCKET')
            collection_name = os.getenv('CB_COLLECTION')
            
            if not all([connection_string, username, password, bucket_name, collection_name]):
                raise ValueError("Missing required Couchbase environment variables")
            
            # Initialize Couchbase connection with timeouts
            auth = PasswordAuthenticator(username, password)
            timeout_options = ClusterTimeoutOptions(
                kv_timeout=timedelta(seconds=5),
                query_timeout=timedelta(seconds=10),
                search_timeout=timedelta(seconds=10)
            )
            options = ClusterOptions(auth, timeout_options=timeout_options)
            
            self.cluster = Cluster(connection_string, options)
            
            # Wait for the cluster to be ready with timeout
            self.cluster.ping()
            
            self.bucket = self.cluster.bucket(bucket_name)
            self.scope = self.bucket.scope("_default")
            self.collection = self.bucket.collection(collection_name)
            self.search_index_name = "kubecontalks._default.kubecontalks"
            
        except Exception as e:
            st.error(f"Failed to initialize Couchbase connection: {str(e)}")
            raise

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the input text using Nebius API"""
        try:
            # Add timeout for embedding generation
            start_time = time.time()
            response = client.embeddings.create(
                model="intfloat/e5-mistral-7b-instruct",
                input=text,
                timeout=30  # 30 seconds timeout
            )
            end_time = time.time()
            st.write(f"Embedding generation took {end_time - start_time:.2f} seconds")
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error generating embedding: {str(e)}")
            raise

    def get_similar_talks(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Perform vector search to find similar talks"""
        try:
            # Generate embedding for the query
            st.write("Generating embedding for query...")
            embedding = self.generate_embedding(query)
            st.write(f"Generated embedding with length: {len(embedding)}")
            
            # Create search request with vector search
            st.write("Creating search request...")
            search_req = SearchRequest.create(MatchNoneQuery()).with_vector_search(
                VectorSearch.from_vector_query(
                    VectorQuery("embedding", embedding, num_candidates=num_results)
                )
            )
            
            # Execute search with timeout
            st.write("Executing vector search...")
            start_time = time.time()
            
            try:
                # First check if the index exists
                indexes = self.cluster.search_indexes().get_all_indexes()
                st.write(f"Available search indexes: {[idx.name for idx in indexes]}")
                
                if not any(idx.name == self.search_index_name for idx in indexes):
                    st.error(f"Search index '{self.search_index_name}' not found!")
                    return []
                
                # Execute the search
                result = self.scope.search(self.search_index_name, search_req)
                
                # Check if we got any results
                if not result:
                    st.warning("No results returned from vector search")
                    return []
                
                # Get the rows
                rows = list(result.rows())
                st.write(f"Found {len(rows)} results")
                
                end_time = time.time()
                st.write(f"Vector search took {end_time - start_time:.2f} seconds")
                
                # Process results
                similar_talks = []
                for row in rows:
                    try:
                        st.write(f"Processing document ID: {row.id}")
                        doc = self.collection.get(row.id, timeout=timedelta(seconds=5))
                        if doc and doc.value:
                            talk = doc.value
                            similar_talks.append({
                                "title": talk.get("title", ""),
                                "description": talk.get("description", ""),
                                "category": talk.get("category", ""),
                                "speaker": talk.get("speaker", ""),
                                "score": row.score
                            })
                            st.write(f"Successfully processed document: {talk.get('title', '')}")
                    except Exception as doc_error:
                        st.warning(f"Error fetching document {row.id}: {str(doc_error)}")
                        continue
                
                return similar_talks
                
            except Exception as search_error:
                st.error(f"Error during vector search: {str(search_error)}")
                return []
            
        except Exception as e:
            st.error(f"Error in get_similar_talks: {str(e)}")
            return []

def generate_talk_suggestion(query: str, similar_talks: List[Dict[str, Any]]) -> str:
    """Generate talk suggestion using OpenAI based on similar talks"""
    try:
        if not similar_talks:
            return "No similar talks found to base the suggestion on. Please try a different query."
            
        # Prepare context from similar talks
        context = "\n\n".join([
            f"Title: {talk['title']}\n"
            f"Description: {talk['description']}\n"
            f"Category: {talk['category']}\n"
            f"Speaker: {talk['speaker']}"
            for talk in similar_talks
        ])

        # Create the prompt
        prompt = f"""You are an expert conference program advisor specializing in cloud-native technologies. 
        Based on the following existing talks and the user's query, create a new talk proposal that:
        1. Focuses on end-user/consumer perspective
        2. Builds upon existing concepts rather than repeating them
        3. Follows a similar structure to successful talks
        4. Addresses current trends and gaps in the topic area

        User Query: {query}

        Similar Existing Talks:
        {context}

        Please provide:
        1. A compelling title
        2. A detailed abstract (2-3 paragraphs)
        3. Key learning objectives
        4. Target audience
        5. Why this talk is unique and valuable
        6. How it builds upon existing knowledge

        Format the response in a clear, structured way with appropriate headings."""

        st.write("Generating talk suggestion...")
        start_time = time.time()
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            messages=[
                {"role": "system", "content": "You are a helpful conference program advisor with expertise in cloud-native technologies and open source."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500,
            timeout=60  # 60 seconds timeout
        )
        end_time = time.time()
        st.write(f"Chat completion took {end_time - start_time:.2f} seconds")
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating suggestion: {str(e)}")
        return f"Error generating suggestion: {str(e)}"

def main():
    st.title("Cloud Native Talk Proposal Generator")
    st.write("Get AI-powered suggestions for your talk proposals based on similar existing talks!")

    try:
        # Initialize Couchbase connection
        with st.spinner("Connecting to Couchbase..."):
            cb = CouchbaseConnection()
        
        # User query input
        user_query = st.text_area(
            "What kind of talk are you interested in proposing?",
            placeholder="E.g., I want to propose a talk about OpenTelemetry's inferred spans feature, focusing on end-user implementation experiences..."
        )

        if st.button("Generate Talk Proposal"):
            if not user_query:
                st.warning("Please enter your query first!")
                return

            with st.spinner("Searching for similar talks and generating proposal..."):
                # Get similar talks
                similar_talks = cb.get_similar_talks(user_query)
                
                if similar_talks:
                    # Display similar talks
                    st.subheader("Similar Existing Talks")
                    for talk in similar_talks:
                        with st.expander(f"{talk['title']} - {talk['speaker']} (Relevance: {talk['score']:.2f})"):
                            st.write("**Description:**")
                            st.write(talk['description'])
                            st.write("**Category:**", talk['category'])
                    
                    # Generate and display new talk proposal
                    st.subheader("Generated Talk Proposal")
                    proposal = generate_talk_suggestion(user_query, similar_talks)
                    st.markdown(proposal)
                else:
                    st.warning("No similar talks found. Please try a different query.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your connection settings and try again.")

if __name__ == "__main__":
    main() 
