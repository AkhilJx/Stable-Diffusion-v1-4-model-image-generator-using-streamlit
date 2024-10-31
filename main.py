import streamlit as st
from diffusers import DiffusionPipeline
import torch
from io import BytesIO

# Load the model for CPU
@st.cache_resource
def load_pipeline():
    pipe = DiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.float32,
        use_safetensors=False,
    ).to("cpu")

    # Optionally, disable safety checkers if not needed
    pipe.safety_checker = None

    return pipe

# Load the model only once
pipe = load_pipeline()

# Streamlit app UI
st.title("Stable Diffusion Image Generator")
st.write("Enter a prompt below to generate an image:")

# Prompt input with suggestion
prompt = st.text_input(
    "Prompt",
    placeholder="Enter your prompt, e.g., A dog in space wearing cooling glasses"
)

# Generate image on button click
if st.button("Generate Image"):
    if prompt:
        with st.spinner("Generating image..."):
            # Generate the image
            image = pipe(prompt=prompt).images[0]

            # Display the image in Streamlit
            st.image(image, caption="Generated Image", use_column_width=True)

            # Save the image to a buffer for download
            buf = BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)

            # Provide a download button for the image
            st.download_button(
                label="Save Image",
                data=buf,
                file_name="generated_image.png",
                mime="image/png"
            )
    else:
        st.warning("Please enter a prompt.")

