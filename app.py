import streamlit as st
import torch
from diffusers import KandinskyPriorPipeline, KandinskyPipeline
from PIL import Image
import io
import os
from huggingface_hub import snapshot_download

# Set page config
st.set_page_config(
    page_title="Kandinsky Text-to-Image Generator",
    page_icon="🎨",
    layout="wide"
)

# Title and description
st.title("🎨 Kandinsky Text-to-Image Generator")
st.markdown("""
    Generate beautiful images from text descriptions using the Kandinsky model.
    Simply enter your prompt and click generate!
""")

# Initialize the models
@st.cache_resource
def load_models():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load prior pipeline
        prior_pipeline = KandinskyPriorPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1-prior",
            torch_dtype=torch_dtype
        ).to(device)
        
        # Load main pipeline
        pipeline = KandinskyPipeline.from_pretrained(
            "kandinsky-community/kandinsky-2-1",
            torch_dtype=torch_dtype
        ).to(device)
        
        return prior_pipeline, pipeline
    except Exception as e:
        st.error(f"Detailed error during model loading: {str(e)}")
        raise

# Load the models
try:
    with st.spinner("Loading models... This might take a few minutes on first run."):
        prior_pipeline, pipeline = load_models()
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

# Create the input form
with st.form("generation_form"):
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
    
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider(
            "Image Width",
            min_value=512,
            max_value=1024,
            value=768,
            step=64
        )
    with col2:
        height = st.slider(
            "Image Height",
            min_value=512,
            max_value=1024,
            value=768,
            step=64
        )
    
    num_inference_steps = st.slider(
        "Number of inference steps",
        min_value=20,
        max_value=100,
        value=50,
        step=1
    )
    
    guidance_scale = st.slider(
        "Guidance scale",
        min_value=1.0,
        max_value=20.0,
        value=7.5,
        step=0.1
    )
    
    generate_button = st.form_submit_button("Generate Image")

# Generate image when the form is submitted
if generate_button and prompt:
    with st.spinner("Generating image..."):
        try:
            # Step 1: Generate image embeddings using the prior pipeline
            image_embeds, negative_image_embeds = prior_pipeline(
                prompt,
                negative_prompt if negative_prompt else None,
                guidance_scale=guidance_scale
            ).to_tuple()
            
            # Step 2: Generate the image using the main pipeline
            image = pipeline(
                prompt,
                image_embeds=image_embeds,
                negative_prompt=negative_prompt if negative_prompt else None,
                negative_image_embeds=negative_image_embeds,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
            
            # Display the image
            st.image(image, caption="Generated Image", use_column_width=True)
            
            # Add download button
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="generated_image.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Error generating image: {str(e)}")
else:
    st.info("👆 Enter a prompt and click 'Generate Image' to create your masterpiece!")

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Kandinsky") 