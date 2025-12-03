import streamlit as st
import ollama 
from PIL import Image 
import io 
import base64

st.set_page_config(
    page_icon="🔎",
    page_title="llava:7b",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    # <img src="data:image/png;base64,{}" width="50" style="vertical-align: -12px;"> Llava:7b OCR
""".format(base64.b64encode(open("./images/ollama.png", "rb").read()).decode()), unsafe_allow_html=True)

col1, col2 = st.columns([6,1])
with col2:
    if st.button("Clear"):
        if 'ocr_result' in st.session_state:
            del st.session_state['ocr_result']
        st.rerun()

st.markdown('<p style="margin-top: -20px;">Extract structured text from images using Llava Vision!</p>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        if st.button("Extract Text 🔍", type="primary"):
            with st.spinner("Processing Image...."):
                try:
                    st.write("Starting OCR process...")
                    
                    # Convert image to bytes
                    img_bytes = uploaded_file.getvalue()
                    st.write(f"Image size: {len(img_bytes)} bytes")
                    
                    st.write("Sending to Ollama...")
                    response = ollama.chat(
                        model='llava:7b',
                        messages=[{
                            'role':'user',
                            'content': """Analyze the text in the provided image. Extract all readable content
                                        and present it in a structured Markdown format that is clear, concise, 
                                        and well-organized. Ensure proper formatting (e.g., headings, lists, or
                                        code blocks) as necessary to represent the content effectively.""",
                            'images': [img_bytes]        
                        }]
                    )
                    
                    st.write("Got response from Ollama!")
                    st.session_state['ocr_result'] = response['message']['content']
                    st.success("Text extracted successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

if 'ocr_result' in st.session_state:
    st.markdown(st.session_state['ocr_result'])
else:
    st.info("Upload an image & click 'Extract Text' to see the results here.")

st.markdown("---")
st.markdown("Made with ❤️ using Llava:7b Vision Model")
