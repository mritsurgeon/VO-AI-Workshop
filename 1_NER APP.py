
from tika import parser
import spacy
from collections import defaultdict
import streamlit as st
import pandas as pd # Keep pandas as Plotly Express might implicitly use it or benefit from it
import plotly.express as px
from gliner_spacy.pipeline import GlinerSpacy
import time
import torch

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

device = get_device()

# Initialize spaCy and add GlinerSpacy
nlp = spacy.blank("en")
# Consider adjusting max_length based on typical document sizes and available memory
nlp.max_length = 2000000
custom_spacy_config = {
    "gliner_model": "urchade/gliner_multi_pii-v1",
    "chunk_size": 250,
    "labels": ["person", "organization", "phone number", "address", "passport number", "email",
               "credit card number", "social security number", "health insurance id number",
               "date of birth", "mobile phone number", "bank account number"],
    "style": "ent",
    "threshold": 0.3,
    "map_location": device
}
nlp.add_pipe("gliner_spacy", config=custom_spacy_config)

def detect_entities(text):
    # Minor optimization: Directly create the list comprehension if no chunking needed
    if len(text) <= nlp.max_length:
        doc = nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]
    else:
        # Handle chunking for very large texts
        chunks = [text[i:i+nlp.max_length] for i in range(0, len(text), nlp.max_length)]
        entities = []
        for chunk in chunks:
            doc = nlp(chunk)
            entities.extend([(ent.text, ent.label_) for ent in doc.ents])
        return entities


def process_file(uploaded_file):
    # Simplified data structures: directly store results instead of nesting under filename
    # file_data = defaultdict(list) # Removed complex nesting
    entity_counts = defaultdict(int)
    entities_dict = defaultdict(list)
    content = ""
    metadata = {}
    start_time = time.time()

    with st.spinner(f'Processing {uploaded_file.name}...'):
        try:
            # Use uploaded_file directly which acts like a file object
            parsed = parser.from_buffer(uploaded_file.read())
            content = parsed.get('content', '') # Use .get for safer access
            metadata = parsed.get('metadata', {})

            if not content:
                st.warning("No text content found in the file.")
                return None, None, None, None # Return None for all expected values

            if len(content) > 10000000:
                st.warning(f"File is large ({len(content)} characters). Processing first 10 million characters.")
                content = content[:10000000]

            # Perform entity detection
            detected_entities = detect_entities(content)

            # Populate entity counts and dictionary in a single loop
            for entity_text, label in detected_entities:
                entity_counts[label] += 1
                entities_dict[label].append(entity_text)


        except Exception as e:
            st.error(f"Error processing file {uploaded_file.name}: {str(e)}")
            return None, None, None, None # Return None on error

    end_time = time.time()
    processing_time = end_time - start_time
    st.success(f"Processing complete for {uploaded_file.name}. Time: {processing_time:.2f} seconds")
    # Return the processed data directly
    return content, entities_dict, metadata, entity_counts

def create_colored_label(label):
    # Standardized label keys to uppercase for reliable lookup
    label_upper = label.upper()
    label_colors = {
        "PERSON": ("#FF4136", "#FFEEEE"),
        "LOCATION": ("#2ECC40", "#EEFFEE"), # Note: LOCATION is not in your default labels list
        "ORGANIZATION": ("#0074D9", "#EEF6FF"),
        "PHONE NUMBER": ("#FF851B", "#FFF6EE"),
        "ADDRESS": ("#B10DC9", "#F9EEFF"),
        "EMAIL": ("#39CCCC", "#EEFFFF"),
        "PASSPORT NUMBER": ("#FFDC00", "#FFFFEE"),
        "CREDIT CARD NUMBER": ("#F012BE", "#FFEEFF"),
        "SOCIAL SECURITY NUMBER": ("#3D9970", "#EEFFF5"),
        "HEALTH INSURANCE ID NUMBER": ("#85144b", "#FFEEF5"),
        "DATE OF BIRTH": ("#7FDBFF", "#EEF9FF"),
        "MOBILE PHONE NUMBER": ("#01FF70", "#EEFFEE"),
        "BANK ACCOUNT NUMBER": ("#001f3f", "#EEF0F5")
    }
    # Get colors, defaulting to grey if label not found
    label_color, bg_color = label_colors.get(label_upper, ("#AAAAAA", "#F5F5F5"))
    label_style = f'background-color: {label_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; margin-left: 5px;' # Added margin
    entity_style = f'background-color: {bg_color}; padding: 1px 3px; border-radius: 3px;' # Added padding/radius
    # Return the original label casing for display if needed, or uppercase
    return label_style, entity_style, label # Use original label for display text

# Streamlit UI
st.title('Data Classification Tool')
st.info(f"Using device: {device}")

uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'doc', 'docx'])

# Clear previous results when a new file is uploaded or removed
if 'entities_dict' not in st.session_state or st.session_state.get('current_file_name') != (uploaded_file.name if uploaded_file else None):
    st.session_state.content = None
    st.session_state.entities_dict = None
    st.session_state.metadata = None
    st.session_state.entity_counts = None
    st.session_state.current_file_name = None


if uploaded_file is not None:
    # Process only if the file hasn't been processed or is different
    if st.session_state.get('current_file_name') != uploaded_file.name:
        st.session_state.current_file_name = uploaded_file.name
        # Call the updated process_file function
        content, entities_dict, metadata, entity_counts = process_file(uploaded_file)

        # Store results in session state only if processing was successful
        if content is not None:
            st.session_state.content = content
            st.session_state.entities_dict = entities_dict
            st.session_state.metadata = metadata
            st.session_state.entity_counts = entity_counts
        else:
            # Ensure session state is cleared if processing failed
             st.session_state.content = None
             st.session_state.entities_dict = None
             st.session_state.metadata = None
             st.session_state.entity_counts = None
             st.session_state.current_file_name = None # Reset filename if processing failed


# Display results if they exist in session state
if st.session_state.get('entity_counts'):
    st.subheader('Counts of each PII Entity Type Detected')
    # Sort counts for display
    sorted_entity_counts = dict(sorted(st.session_state.entity_counts.items(), key=lambda item: item[1], reverse=True))
    # Create bar chart using the sorted data
    if sorted_entity_counts:
         fig = px.bar(
             x=list(sorted_entity_counts.values()),
             y=list(sorted_entity_counts.keys()),
             orientation='h',
             title=f'PII Counts for {st.session_state.current_file_name}'
         )
         fig.update_layout(yaxis={'categoryorder':'total ascending'}) # Ensure bars are ordered correctly
         st.plotly_chart(fig, use_container_width=True)
    else:
         st.write("No entities detected to display counts.")


if st.session_state.get('entities_dict'):
    st.subheader(f'Detected Entities in {st.session_state.current_file_name}')
    col1, col2 = st.columns(2)
    # Iterate directly through the entities_dict stored in session state
    # Sort items by label for consistent display order
    sorted_entities = sorted(st.session_state.entities_dict.items())
    for i, (label, entities) in enumerate(sorted_entities):
        column = col1 if i % 2 == 0 else col2
        with column:
            # Display the label type once
            st.markdown(f"**{label}** ({len(entities)})")
            # Use an expander for potentially long lists of entities
            with st.expander("Show/Hide Entities", expanded=False):
                for entity in entities:
                    label_style, entity_style, display_label = create_colored_label(label)
                    # Basic HTML escaping for entity text to prevent injection issues
                    import html
                    escaped_entity = html.escape(entity)
                    entity_lines = escaped_entity.split('\n')
                    # Apply style to each line and join with <br>
                    entity_html = '<br>'.join([f'<span style="{entity_style}">{line}</span>' for line in entity_lines])
                    # Display entity and its label tag
                    st.markdown(f'{entity_html} <span style="{label_style}">{display_label}</span>', unsafe_allow_html=True)
            st.write("") # Add spacing between types

