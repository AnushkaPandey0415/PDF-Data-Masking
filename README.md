# PDF Data Masking Tool

## Overview
This web application is designed to identify and mask sensitive data in PDF documents, including both text and images. The tool ensures privacy by detecting and redacting personal information across multiple languages and formats.

## Features
- **Text and Image Processing**: Extracts text from PDFs and images within PDFs for analysis.
- **Multi-Language Name Identification**: Detects names in English, Chinese, Korean, Arabic, and Malay using an advanced Named Entity Recognition (NER) model.
- **Phone Number Masking**: Identifies and masks phone numbers in standard, domestic, and international formats.
- **Email Redaction**: Detects and masks email addresses.
- **Hospital & Clinic Name Masking**: Recognizes and masks medical facility names.

## Technologies Used
### Libraries & Tools
- **PyMuPDF**: Extracts text from PDF documents.
- **Pytesseract OCR**: Extracts text from images within PDFs and determines the coordinates of text to be masked.
- **PIL (Pillow)**: Applies image processing filters to enhance OCR accuracy.
- **OpenCV**: Identifies text pixels in images and applies masking to redact sensitive information.
- **Regular Expressions (re)**: Identifies email addresses, phone numbers, and mentions of hospitals/clinics.
- **XLM-Roberta NER Model**: Utilizes an advanced Named Entity Recognition (NER) model to detect names in multiple languages.

## How It Works
1. **Text Extraction**:
   - PyMuPDF extracts text from the PDF document.
   - Pytesseract OCR is used for text within images.
2. **Data Detection**:
   - Regular expressions identify phone numbers, email addresses, and hospital/clinic names.
   - XLM-Roberta NER model detects names in various languages.
3. **Redaction Process**:
   - OpenCV and PIL mask detected text in both images and standard PDF text.
   - The modified document is then saved with redacted data.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/pdf-data-masking.git
   cd pdf-data-masking
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure Tesseract OCR is installed and configured:
   - Download and install Tesseract OCR from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract)
   - Set the Tesseract path in your environment variables.
4. Run the application:
   ```bash
   python main.py
   ```

## Usage
1. Upload a PDF file.
2. The tool scans and detects sensitive information.
3. Redacted PDF is generated and available for download.

## Future Enhancements
- Integration with cloud storage services.
- Improved UI for better user experience.
- Expansion to support additional languages and entity types.

## License
This project is licensed under the MIT License.

## Contributors
- kaush7040
- pandeyanushka0415

