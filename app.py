import requests
import streamlit as st

api_key = st.secrets["api_key"]

API_URL = "https://api-inference.huggingface.co/models/flair/ner-english-large"
headers = {"Authorization": f"Bearer {api_key}"}


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


def get_ner_from_transformer(output):
    data = output
    named_entities = {}
    for entity in data:
        entity_type = entity["entity_group"]
        entity_text = entity["word"]

        if entity_type not in named_entities:
            named_entities[entity_type] = []

        named_entities[entity_type].append(entity_text)

    return named_entities


def main():
    st.title("Named Entity Recognition with Transformer Demo")

    text = st.text_area("Enter some text to detect named entities")

    if st.button("Detect Named Entities"):
        if text:
            output = query({"inputs": text})
            named_entities = get_ner_from_transformer(output)

            st.subheader("Named Entities Detected:")
            if named_entities:
                for entity_type, entities in named_entities.items():
                    st.write(f"- {entity_type}: {', '.join(entities)}")
            else:
                st.write("No named entities detected.")

        else:
            st.warning("Please enter some text before detecting named entities.")


if __name__ == "__main__":
    main()
