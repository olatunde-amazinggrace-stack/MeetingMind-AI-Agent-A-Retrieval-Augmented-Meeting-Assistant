
import streamlit as st

st.set_page_config(page_title="Custom CSS Example", layout="wide")
st.title("🎨 Custom CSS in Streamlit")

st.markdown("""
    This app demonstrates how to apply custom CSS for advanced styling.
    """)

# Inject custom CSS
st.markdown("""
    <style>
    .stApp { /* Targets the main app container */
        background-color: #f0f2f6; /* Light gray background */
    }
    h1 { /* Styles all H1 headers */
        color: #4CAF50; /* Green color */
        text-align: center;
    }
    .stButton>button { /* Styles all buttons */
        background-color: #008CBA;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stTextInput>div>div>input { /* Styles text input fields */
        border-radius: 8px;
        border: 2px solid #008CBA;
    }
    </style>
    """, unsafe_allow_html=True)

st.header("Styled Elements Below")
st.write("This text input and button are styled using custom CSS.")

name = st.text_input("Enter your name")
if st.button("Submit"): 
    st.success(f"Hello, {name}!")
