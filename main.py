import streamlit as st #fronntend
import fitz  # also called PyMuPDwF
import pytesseract #OCR
from PIL import Image #extract text from image
import cv2 #open image
import numpy as np
import io
import re 
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Set the model directory path
model_name = "C:\\Code\\PDF Masking\\saved_model" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Set the path for Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

patterns = {
    'phone_numbers1': r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
    'phone_numbers2': r'\+\d{2} \d{2}-\d{4}-\d{4}\b',
    'phone_numbers3': r'\+\d{12}\b',
    'phone_numbers4': r'\+\d{2}-\d{2}-\d{4}-\d{4}\b',
    'phone_numbers5': r'\d{2}-\d{4}-\d{4}\b',
    'emails': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'clinics': r'\bClinic\b',
    'hospitals': r'\bHospital\b',
    'clinic_hospital': r'\b(\w+)\s+(Clinic|Hospital)\b', 
    'names_chinese': r'[\u4e00-\u9fa5]{2,}', 
    'names_korean': r'[\uac00-\ud7a3]{2,}', 
    'names_arabic': r'[\u0600-\u06FF]{2,}',  
}

compiled_patterns = {key: re.compile(pattern, re.IGNORECASE) for key, pattern in patterns.items()}

def mask_text_in_image(image, patterns): 
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detailed_data = pytesseract.image_to_data(gray_image, lang='ara+chi_sim+chi_tra+eng+kor+msa', output_type=pytesseract.Output.DICT)

    n_boxes = len(detailed_data['text'])
    word_info = []
    
    for i in range(n_boxes):
        word = detailed_data['text'][i]
        if word.strip():
            x, y, w, h = detailed_data['left'][i], detailed_data['top'][i], detailed_data['width'][i], detailed_data['height'][i]
            word_info.append((word, x, y, w, h))

    for i, (word, x, y, w, h) in enumerate(word_info):
        if any(pattern.search(word) for pattern in compiled_patterns.values()):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)
            if i > 0:
                prev_word, prev_x, prev_y, prev_w, prev_h = word_info[i - 1]
                cv2.rectangle(image, (prev_x, prev_y), (prev_x + prev_w, prev_y + prev_h), (0, 0, 0), -1)

    return image

def extract_text_from_image(image, lang='eng'):
    text = pytesseract.image_to_string(image, lang=lang)
    return text

def replace_image_in_pdf(page, rect, masked_image):
    masked_image_pil = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    masked_image_bytes = io.BytesIO()
    masked_image_pil.save(masked_image_bytes, format='PNG')
    masked_image_bytes = masked_image_bytes.getvalue()
    page.insert_image(rect, stream=masked_image_bytes)

def update_patterns_with_names(page):
    page_text = page.get_text("text")
    entities = ner_pipeline(page_text)

    names = []
    for entity in entities:
        if entity['entity_group'] == 'PER':
            name_parts = entity['word'].split()
            if any(char.isdigit() for char in str(name_parts)):
                pass
            else:
                names.extend(name_parts)
    
    names_pattern = r'\b(' + '|'.join(re.escape(name) for name in names) + r')\b'

    patterns.update({
        'names': names_pattern,
    })

    compiled_patterns.update({'names': re.compile(names_pattern, re.IGNORECASE)})

def mask_sensitive_info(text, patterns):
    for pattern_name, pattern in patterns.items():
        if pattern.search(text):
            return '[MASKED]'
    return text

def mask_text_at_word_level(input_pdf_path, output_pdf_path, font_path):
    pdf_document = fitz.open(input_pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        update_patterns_with_names(page)
        
        words = page.get_text("words")
        words_to_mask = []
        
        for i, word in enumerate(words):
            x0, y0, x1, y1, text, block_number, line_number, span_number = word
            if any(pattern.search(text) for pattern in compiled_patterns.values()):
                words_to_mask.append((text, x0, y0, x1, y1))
                
                if re.search(r'\bClinic\b|\bHospital\b', text, re.IGNORECASE):
                    if i > 0:
                        prev_word = words[i - 1]
                        prev_x0, prev_y0, prev_x1, prev_y1, prev_text, _, _, _ = prev_word
                        words_to_mask.append((prev_text, prev_x0, prev_y0, prev_x1, prev_y1))
        
        for text, x0, y0, x1, y1 in words_to_mask:
            rect = fitz.Rect(x0, y0, x1, y1)
            page.add_redact_annot(rect, fill=(0, 0, 0))
            page.apply_redactions()
            masked_text = mask_sensitive_info(text, compiled_patterns)
            page.insert_text(rect.bl, masked_text, fontsize=12, fontname="helv", fontfile=font_path, color=(1, 1, 1), render_mode=3)
        
        images = page.get_images(full=True)
        for img_index, img_info in enumerate(images):
            xref = img_info[0]
            try:
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                text_from_image = extract_text_from_image(image_cv, lang='ara+chi_sim+chi_tra+eng+kor+msa')
                
                entities = ner_pipeline(text_from_image)
                names = []
                for entity in entities:
                    if entity['entity_group'] == 'PER':
                        name_parts = entity['word'].split()
                        names.extend(name_parts)
                names_pattern = r'\b(' + '|'.join(re.escape(name) for name in names) + r')\b'
                patterns.update({
                    'names': names_pattern,
                })
                compiled_patterns.update({'names': re.compile(names_pattern, re.IGNORECASE)})
                
                masked_image = mask_text_in_image(image_cv, compiled_patterns)

                rect = page.get_image_rects(xref)[0]

                if (rect.x0 < rect.x1) and (rect.y0 < rect.y1):
                    replace_image_in_pdf(page, rect, masked_image)
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    pdf_document.save(output_pdf_path)

def main():
    st.set_page_config(page_title="PDF Data Masking Application", page_icon="🛡️", layout="wide")
    st.markdown('<h1 style="color: yellow;">🛡️ PDF Data Masking Application 🛡️</h1>', unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .main {
            background-color: #333333;
            padding: 20px;
            border-radius: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="about">
            <h2 style="color: white;">About This Application</h2>
            <p style="color: #cccccc;">
                This web page can mask both text and images inside pdfs to identify and mask the following data:
            </p>
            <ul style="color: #cccccc;">
                <li><strong>Names:</strong> Masks English, Chinese, Korean, Arabic and Malay names.</li>
                <li><strong>Phone Numbers:</strong> Masks phone numbers in standard, domestic and international formats.</li>
                <li><strong>Emails:</strong> Masks email addresses in any format.</li>
                <li><strong>Hospitals and Clinic Names:</strong> Masks names of hospitals and clinics.</li>
            </ul>
            <p style="color: #cccccc;">
                Simply upload your PDF, and our application will handle the rest, ensuring that all identified sensitive data is masked effectively!
            </p>
        </div>
        """, unsafe_allow_html=True
    )


    st.sidebar.header("Upload PDF")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner("Processing..."):
            os.makedirs("uploads", exist_ok=True)
            os.makedirs("outputs", exist_ok=True)

            input_pdf_path = os.path.join("uploads", uploaded_file.name)
            with open(input_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            output_pdf_path = os.path.join("outputs", "masked_" + uploaded_file.name)
            font_file_path = r"C:\Windows\Fonts\arial.ttf"  # Path to font file

            mask_text_at_word_level(input_pdf_path, output_pdf_path, font_file_path)

            st.success("PDF processing complete.")
            st.balloons()
            st.download_button("Download Masked PDF", open(output_pdf_path, "rb"), file_name="masked_" + uploaded_file.name)

if __name__ == "__main__":
    main()
