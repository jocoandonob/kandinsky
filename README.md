# Kandinsky Text-to-Image Generator

A web application that generates images from text descriptions using the Kandinsky 2.2 model. Built with Streamlit and Python.

## Features

- Text-to-image generation using Kandinsky 2.2
- Customizable generation parameters
- Negative prompt support
- Image download functionality
- User-friendly interface

## Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd kandinsky
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

3. Enter your text prompt and adjust the generation parameters as needed

4. Click "Generate Image" to create your image

5. Download the generated image using the download button

## Parameters

- **Prompt**: The text description of the image you want to generate
- **Negative Prompt**: Elements you want to avoid in the generated image
- **Number of Inference Steps**: Higher values may produce better quality but take longer
- **Guidance Scale**: Controls how closely the image follows the prompt (higher values = more adherence)

## Notes

- The first run will download the model weights, which may take some time
- GPU is recommended for faster generation
- Generation time depends on your hardware and the number of inference steps

## License

This project is licensed under the MIT License - see the LICENSE file for details. 