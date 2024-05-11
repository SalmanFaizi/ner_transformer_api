import requests
import streamlit as st

api_key = st.secrets["api_key"]

API_URL = "https://api-inference.huggingface.co/models/flair/ner-english-large"
headers = {"Authorization": "Bearer {api_key}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def get_ner_from_transformer(output):
    data = output
    named_entities = {}
    for entity in data:
        entity_type = entity['entity_group']
        entity_text = entity['word']
        
        if entity_type not in named_entities:
            named_entities[entity_type] = []
        
        named_entities[entity_type].append(entity_text)
    
    return entity_type, named_entities

text=st.text_area("enter some text to detect ner")
output = query({
    "inputs": text,
})

entity_type, named_entities = get_ner_from_transformer(output)
st.write(entity_type)
st.write(named_entities)
