import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pytesseract
from typing import List, Tuple

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = "PiyushTiwari10/OCR/tesseract.exe"

# Extract text from image using Tesseract OCR
def extract_text(image: Image.Image) -> str:
    try:
        text = pytesseract.image_to_string(image)
        return text
    except pytesseract.TesseractError as e:
        st.error(f"An error occurred while extracting text: {e}")
        return "Text extraction failed. Please try again."
    except Exception as e:
        st.error(f"An unexpected error occurred while extracting text: {e}")
        return "Text extraction failed. Please try again."

# Segment visual elements using OpenCV
def segment_visual_elements(image_path: str) -> Tuple[np.ndarray, List[np.ndarray]]:
    try:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and aspect ratio
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                filtered_contours.append(contour)
        
        return image, filtered_contours
    except Exception as e:
        st.error(f"An error occurred while segmenting visual elements: {e}")
        return None, []

# Save segmented visual elements as separate images
def save_visual_elements(image: np.ndarray, contours: List[np.ndarray], output_dir: str) -> List[str]:
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        element_paths = []
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            visual_element = image[y:y+h, x:x+w]
            element_path = os.path.join(output_dir, f'element_{i}.png')
            cv2.imwrite(element_path, visual_element)
            element_paths.append(element_path)
        return element_paths
    except Exception as e:
        st.error(f"An error occurred while saving visual elements: {e}")
        return []

# Generate HTML content
def generate_html(text: str, element_paths: List[str]) -> str:
    html_content = '<html><body>'
    html_content += f'<p>{text}</p>'
    for path in element_paths:
        html_content += f'<img src="{path}" />'
    html_content += '</body></html>'
    return html_content

def main():
    st.title("Image Text and Visual Element Extractor")

    # Explanation of the app
    st.write("""
    This web application allows you to extract text and visual elements from images. 
    It utilizes Optical Character Recognition (OCR) to extract text content from the uploaded image 
    and basic image segmentation techniques to isolate and display individual visual elements found within the image.
    """)

    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    sidebar_image_path = "alien.png"
    st.sidebar.image(sidebar_image_path, caption="", use_column_width=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_path = f"temp_{uploaded_file.name}"
        image.save(image_path)

        # Display uploaded image in two columns
        col1, col2 = st.columns([2, 3])
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_column_width=True)

        # Processing with a spinner
        with col2:
            with st.spinner("🤖 Extracting Text..."):
                text = extract_text(image)
                st.subheader("Extracted Text")
                st.write(text)

                # Add a button to copy the extracted text
                st.code(text, language="text")
                # st.markdown(f'<button onclick="navigator.clipboard.writeText(`{text}`)">Copy Text</button>', unsafe_allow_html=True)

                # Segment visual elements
                image_cv2, contours = segment_visual_elements(image_path)
                element_paths = save_visual_elements(image_cv2, contours, "output_elements")

                st.subheader("Visual Elements")
                for path in element_paths:
                    st.image(path, use_column_width=True)

                html_content = generate_html(text, element_paths)
                st.download_button("Download HTML", html_content, "output.html", "text/html")

        # Cleanup
        os.remove(image_path)
        for path in element_paths:
            os.remove(path)

if __name__ == "__main__":
    main()
