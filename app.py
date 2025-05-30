import streamlit as st
import torch
from diffusers import KandinskyPriorPipeline, KandinskyPipeline, KandinskyImg2ImgPipeline, KandinskyInpaintPipeline
from PIL import Image
import io
import os
import requests
from io import BytesIO
import numpy as np
from diffusers.utils import load_image, make_image_grid

# Set page config
st.set_page_config(
    page_title="Kandinsky Image Generator",
    page_icon="🎨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Form container styling */
    .stForm {
        background-color: #1E1E1E;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        height: 100px !important;
        background-color: #2D2D2D !important;
        color: #FFFFFF !important;
        border: 1px solid #444 !important;
        border-radius: 5px !important;
    }
    
    /* Button styling */
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.8rem 1.5rem;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #1E1E1E;
        padding: 0.5rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        white-space: pre-wrap;
        background-color: #2D2D2D;
        border-radius: 5px;
        gap: 1rem;
        padding: 0.5rem 1.5rem;
        color: #FFFFFF;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #3D3D3D;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Slider styling */
    .stSlider {
        padding: 1rem 0;
    }
    
    .stSlider [data-baseweb="slider"] {
        background-color: #2D2D2D;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: #2D2D2D;
        border-radius: 5px;
        padding: 1rem;
        border: 1px solid #444;
    }
    
    /* Image container styling */
    .image-container {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
    }
    
    /* Title styling */
    h1 {
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: 600;
    }
    
    /* Subtitle styling */
    h3 {
        color: #FFFFFF;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 500;
    }
    
    /* Info message styling */
    .stInfo {
        background-color: #2D2D2D;
        border: 1px solid #444;
        border-radius: 5px;
        padding: 1rem;
    }
    
    /* Download button styling */
    .stDownloadButton button {
        background-color: #2196F3;
    }
    
    .stDownloadButton button:hover {
        background-color: #1976D2;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Kandinsky Image Generator")
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem; color: #FFFFFF;'>
        Generate and transform images using the Kandinsky model
    </div>
""", unsafe_allow_html=True)

# Initialize the models
@st.cache_resource
def load_models():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load prior pipeline
        prior_pipeline = KandinskyPriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1-prior",
            torch_dtype=torch_dtype,
            use_safetensors=True
        ).to(device)
        
        # Load text-to-image pipeline
        text2img_pipeline = KandinskyPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1",
            torch_dtype=torch_dtype,
            use_safetensors=True
        ).to(device)
        
        # Load image-to-image pipeline
        img2img_pipeline = KandinskyImg2ImgPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1",
            torch_dtype=torch_dtype,
            use_safetensors=True
        ).to(device)
        
        # Load inpainting pipeline
        inpaint_pipeline = KandinskyInpaintPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1-inpaint",
            torch_dtype=torch_dtype,
            use_safetensors=True
        ).to(device)
        
        return prior_pipeline, text2img_pipeline, img2img_pipeline, inpaint_pipeline
    except Exception as e:
        st.error(f"Detailed error during model loading: {str(e)}")
        raise

# Load the models
try:
    with st.spinner("Loading models... This might take a few minutes on first run."):
        prior_pipeline, text2img_pipeline, img2img_pipeline, inpaint_pipeline = load_models()
        st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"""
    Error loading models: {str(e)}
    
    Please try the following:
    1. Check your internet connection
    2. Make sure you have enough disk space (at least 5GB free)
    3. Try clearing your cache:
       - Windows: Delete the folder: %USERPROFILE%\.cache\huggingface
       - Linux/Mac: Delete the folder: ~/.cache/huggingface
    4. If the problem persists, try running these commands in your terminal:
       pip install --upgrade huggingface_hub
       pip install --upgrade diffusers
    """)
    st.stop()

# Create tabs with new styling
tab1, tab2, tab3, tab4 = st.tabs(["Text to Image", "Image to Image", "Inpainting", "Interpolation"])

# Wrap the content in a container
with st.container():
    # Text to Image Tab
    with tab1:
        # Create two columns for the layout
        col1, col2 = st.columns([1, 1])

        # Left column for inputs
        with col1:
            with st.form("text2img_form"):
                prompt = st.text_area(
                    "Enter your prompt",
                    placeholder="A beautiful sunset over mountains, digital art style",
                    height=100
                )
                
                negative_prompt = st.text_area(
                    "Negative prompt (optional)",
                    placeholder="blurry, low quality, distorted",
                    height=100
                )
                
                st.markdown("### Image Settings")
                
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    width = st.slider(
                        "Width",
                        min_value=512,
                        max_value=1024,
                        value=768,
                        step=64
                    )
                with col1_2:
                    height = st.slider(
                        "Height",
                        min_value=512,
                        max_value=1024,
                        value=768,
                        step=64
                    )
                
                st.markdown("### Generation Settings")
                num_inference_steps = st.slider(
                    "Inference Steps",
                    min_value=20,
                    max_value=100,
                    value=50,
                    step=1
                )
                
                guidance_scale = st.slider(
                    "Guidance Scale",
                    min_value=1.0,
                    max_value=20.0,
                    value=7.5,
                    step=0.1
                )
                
                generate_button = st.form_submit_button(" Generate Image")

        # Right column for results
        with col2:
            st.markdown("### Generated Image")
            if 'text2img_image' not in st.session_state:
                st.info("👆 Enter a prompt and click 'Generate Image' to create your masterpiece!")
            else:
                with st.container():
                    st.image(st.session_state.text2img_image, width=500)
                    if 'text2img_bytes' in st.session_state:
                        st.download_button(
                            label="⬇️ Download Image",
                            data=st.session_state.text2img_bytes,
                            file_name="generated_image.png",
                            mime="image/png"
                        )

    # Image to Image Tab
    with tab2:
        # Create two columns for the layout
        col1, col2 = st.columns([1, 1])

        # Left column for inputs
        with col1:
            with st.form("img2img_form"):
                # Image upload
                uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
                
                # URL input
                image_url = st.text_input("Or enter an image URL")
                
                prompt = st.text_area(
                    "Enter your prompt",
                    placeholder="A fantasy landscape, Cinematic lighting",
                    height=100
                )
                
                negative_prompt = st.text_area(
                    "Negative prompt (optional)",
                    placeholder="low quality, bad quality",
                    height=100
                )
                
                st.markdown("### Image Settings")
                
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    width = st.slider(
                        "Width",
                        min_value=512,
                        max_value=1024,
                        value=768,
                        step=64
                    )
                with col1_2:
                    height = st.slider(
                        "Height",
                        min_value=512,
                        max_value=1024,
                        value=768,
                        step=64
                    )
                
                st.markdown("### Generation Settings")
                strength = st.slider(
                    "Transformation Strength",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.3,
                    step=0.1
                )
                
                num_inference_steps = st.slider(
                    "Inference Steps",
                    min_value=20,
                    max_value=100,
                    value=50,
                    step=1
                )
                
                guidance_scale = st.slider(
                    "Guidance Scale",
                    min_value=1.0,
                    max_value=20.0,
                    value=7.5,
                    step=0.1
                )
                
                generate_button = st.form_submit_button("🔄 Transform Image")

        # Right column for results
        with col2:
            st.markdown("### Generated Image")
            if 'img2img_image' not in st.session_state:
                st.info("👆 Upload an image or enter an URL and click 'Transform Image' to create your masterpiece!")
            else:
                with st.container():
                    st.image(st.session_state.img2img_image, width=500)
                    if 'img2img_bytes' in st.session_state:
                        st.download_button(
                            label="⬇️ Download Image",
                            data=st.session_state.img2img_bytes,
                            file_name="transformed_image.png",
                            mime="image/png"
                        )

    # Inpainting Tab
    with tab3:
        # Create two columns for the layout
        col1, col2 = st.columns([1, 1])

        # Left column for inputs
        with col1:
            with st.form("inpaint_form"):
                # Image upload
                uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
                
                # URL input
                image_url = st.text_input("Or enter an image URL")
                
                prompt = st.text_area(
                    "Enter your prompt",
                    placeholder="a hat, a crown, a flower",
                    height=100
                )
                
                negative_prompt = st.text_area(
                    "Negative prompt (optional)",
                    placeholder="low quality, bad quality",
                    height=100
                )
                
                st.markdown("### Image Settings")
                
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    width = st.slider(
                        "Width",
                        min_value=512,
                        max_value=1024,
                        value=768,
                        step=64
                    )
                with col1_2:
                    height = st.slider(
                        "Height",
                        min_value=512,
                        max_value=1024,
                        value=768,
                        step=64
                    )
                
                st.markdown("### Generation Settings")
                num_inference_steps = st.slider(
                    "Inference Steps",
                    min_value=20,
                    max_value=150,
                    value=50,
                    step=1
                )
                
                guidance_scale = st.slider(
                    "Guidance Scale",
                    min_value=1.0,
                    max_value=20.0,
                    value=7.5,
                    step=0.1
                )
                
                generate_button = st.form_submit_button("🎨 Generate Inpainting")

        # Right column for results
        with col2:
            st.markdown("### Generated Image")
            if 'inpaint_image' not in st.session_state:
                st.info("👆 Upload an image and click 'Generate Inpainting' to create your masterpiece!")
            else:
                with st.container():
                    st.image(st.session_state.inpaint_image, width=500)
                    if 'inpaint_bytes' in st.session_state:
                        st.download_button(
                            label="⬇️ Download Image",
                            data=st.session_state.inpaint_bytes,
                            file_name="inpainted_image.png",
                            mime="image/png"
                        )

        # Generate image when the form is submitted
        if generate_button and prompt and (uploaded_file or image_url):
            with st.spinner("Generating inpainting..."):
                try:
                    # Load the input image
                    if uploaded_file:
                        init_image = Image.open(uploaded_file)
                    else:
                        response = requests.get(image_url)
                        init_image = Image.open(BytesIO(response.content))
                    
                    # Resize the input image
                    init_image = init_image.resize((width, height))
                    
                    # Create a simple mask (you can modify this to allow user to draw the mask)
                    mask = np.zeros((height, width), dtype=np.float32)
                    # Example: mask the top portion of the image
                    mask[:height//3, :] = 1
                    
                    # Convert mask to PIL Image
                    mask_image = Image.fromarray((mask * 255).astype('uint8'), 'L')
                    
                    # Generate prior embeddings
                    prior_output = prior_pipeline(
                        prompt,
                        negative_prompt if negative_prompt else None,
                        guidance_scale=guidance_scale
                    )
                    
                    # Generate the inpainted image
                    image = inpaint_pipeline(
                        prompt,
                        image=init_image,
                        mask_image=mask_image,
                        **prior_output,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale
                    ).images[0]
                    
                    # Store the image in session state
                    st.session_state.inpaint_image = image
                    
                    # Prepare image for download
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.session_state.inpaint_bytes = buf.getvalue()
                    
                    # Rerun to update the display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating inpainting: {str(e)}")

    # Interpolation Tab
    with tab4:
        # Create two columns for the layout
        col1, col2 = st.columns([1, 1])

        # Left column for inputs
        with col1:
            with st.form("interpolation_form"):
                st.markdown("### Input Images")
                
                # First image input
                st.markdown("#### First Image")
                img1_upload = st.file_uploader("Upload first image", type=["png", "jpg", "jpeg"], key="img1")
                img1_url = st.text_input("Or enter first image URL", key="url1")
                
                # Second image input
                st.markdown("#### Second Image")
                img2_upload = st.file_uploader("Upload second image", type=["png", "jpg", "jpeg"], key="img2")
                img2_url = st.text_input("Or enter second image URL", key="url2")
                
                # Optional text prompt
                prompt = st.text_area(
                    "Optional text prompt",
                    placeholder="Leave empty for pure image interpolation",
                    height=100
                )
                
                st.markdown("### Interpolation Settings")
                
                # Weight sliders
                st.markdown("#### Weights")
                weight1 = st.slider(
                    "First Image Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1
                )
                
                weight2 = st.slider(
                    "Second Image Weight",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1
                )
                
                # Text weight (remaining weight)
                text_weight = 1.0 - (weight1 + weight2)
                st.markdown(f"Text Weight: {text_weight:.1f}")
                
                st.markdown("### Image Settings")
                
                col1_1, col1_2 = st.columns(2)
                with col1_1:
                    width = st.slider(
                        "Width",
                        min_value=512,
                        max_value=1024,
                        value=768,
                        step=64
                    )
                with col1_2:
                    height = st.slider(
                        "Height",
                        min_value=512,
                        max_value=1024,
                        value=768,
                        step=64
                    )
                
                generate_button = st.form_submit_button("🔄 Generate Interpolation")

        # Right column for results
        with col2:
            st.markdown("### Generated Image")
            if 'interpolation_image' not in st.session_state:
                st.info("👆 Upload two images and click 'Generate Interpolation' to create your masterpiece!")
            else:
                with st.container():
                    st.image(st.session_state.interpolation_image, width=500)
                    if 'interpolation_bytes' in st.session_state:
                        st.download_button(
                            label="⬇️ Download Image",
                            data=st.session_state.interpolation_bytes,
                            file_name="interpolated_image.png",
                            mime="image/png"
                        )

        # Generate image when the form is submitted
        if generate_button and ((img1_upload or img1_url) and (img2_upload or img2_url)):
            with st.spinner("Generating interpolation..."):
                try:
                    # Load the input images
                    def load_input_image(upload, url):
                        if upload:
                            return Image.open(upload)
                        else:
                            response = requests.get(url)
                            return Image.open(BytesIO(response.content))
                    
                    img1 = load_input_image(img1_upload, img1_url)
                    img2 = load_input_image(img2_upload, img2_url)
                    
                    # Resize images
                    img1 = img1.resize((512, 512))
                    img2 = img2.resize((512, 512))
                    
                    # Prepare inputs for interpolation
                    images_texts = [prompt, img1, img2]
                    weights = [text_weight, weight1, weight2]
                    
                    # Generate prior embeddings
                    prior_output = prior_pipeline.interpolate(images_texts, weights)
                    
                    # Generate the interpolated image
                    image = text2img_pipeline(
                        prompt,
                        **prior_output,
                        height=height,
                        width=width
                    ).images[0]
                    
                    # Store the image in session state
                    st.session_state.interpolation_image = image
                    
                    # Prepare image for download
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    st.session_state.interpolation_bytes = buf.getvalue()
                    
                    # Rerun to update the display
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error generating interpolation: {str(e)}")

# Add footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit and Kandinsky
    </div>
""", unsafe_allow_html=True) 