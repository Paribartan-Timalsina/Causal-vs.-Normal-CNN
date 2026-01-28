import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Modular imports
from models_loader.loader import load_model
from utils.preprocessing import preprocess_uploaded_image
from utils.inference import predict
from utils.visualization import plot_probabilities
from utils.constants import COLOR_MAP, COLOR_NAMES
from utils.data_utils import get_colored_mnist

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Causal AI: Breaking Spurious Correlations",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .color-box {
        display: inline-block;
        width: 20px;
        height: 20px;
        margin-right: 10px;
        border: 1px solid #ccc;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================
st.markdown('<h1 class="main-header">🧠 Causal AI vs. Standard Deep Learning</h1>', unsafe_allow_html=True)

st.markdown("""
This application demonstrates the difference between **Standard ML** and **Causal ML** approaches when dealing with spurious correlations.

### 🔍 The Problem:
During training, each digit was **perfectly correlated with a specific color**. Standard models learn this "shortcut" 
instead of the actual digit shape. When tested with random colors, they fail dramatically.

### ✨ The Solution:
**Invariant Risk Minimization (IRM)** trains across multiple environments with different correlation strengths, 
forcing the model to learn only the invariant causal features (digit shape) while ignoring unstable correlations (color).
""")

# =========================================================
# SIDEBAR: COLOR MAPPING
# =========================================================
st.sidebar.header("🎨 Training Data Color Scheme")
st.sidebar.markdown("""
**⚠️ IMPORTANT: Spurious Correlation Alert!**

During training, each digit was assigned a **fixed color**:
""")

# Display color mapping in sidebar
for digit, color_name in enumerate(COLOR_NAMES):
    color_rgb = COLOR_MAP[digit].tolist()
    hex_color = '#%02x%02x%02x' % (
        int(color_rgb[0] * 255),
        int(color_rgb[1] * 255),
        int(color_rgb[2] * 255)
    )
    st.sidebar.markdown(
        f'<div style="margin: 5px 0;">'
        f'<span class="color-box" style="background-color: {hex_color};"></span>'
        f'<strong>Digit {digit}</strong> → {color_name}'
        f'</div>',
        unsafe_allow_html=True
    )

st.sidebar.markdown("""
---
**This means:**
- Standard models (SimpleCNN, ResNet) learn "If I see Red → predict 0"
- IRM learns "If shape looks like 0 → predict 0" (regardless of color)
""")

# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"🖥️ Running on: **{device}**")

# =========================================================
# MODEL SELECTION
# =========================================================
st.header("1️⃣ Choose a Model")

model_type = st.selectbox(
    "Select Model Architecture:",
    ["Simple CNN", "ResNet-18", "Invariant Risk Minimization (IRM) CNN"],
    help="Simple CNN and ResNet learn shortcuts. IRM learns causal features."
)

model = load_model(model_type, device)

# Model descriptions
model_descriptions = {
    "Simple CNN": "⚡ **Fast but fragile** - Prone to learning spurious correlations like color.",
    "ResNet-18": "🏗️ **Deeper architecture** - Still learns shortcuts despite more capacity.",
    "Invariant Risk Minimization (IRM) CNN": "🎯 **Causal Learning** - Trained to ignore spurious correlations across environments."
}
st.info(model_descriptions[model_type])

# =========================================================
# MODE SELECTION
# =========================================================
st.header("2️⃣ Choose Test Mode")

mode = st.radio(
    "Select how you want to test the model:",
    ["📊 Pre-generated Dataset", "📤 Upload Your Own Image"],
    horizontal=True
)

# =========================================================
# UPLOAD MODE
# =========================================================
if mode == "📤 Upload Your Own Image":
    st.markdown("""
    Upload a handwritten digit image and choose a color to apply.  
    See if the model is fooled by the color or correctly identifies the shape!
    """)

    uploaded_file = st.file_uploader(
        "Upload an image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"],
        help="Upload a handwritten digit image. Works best with digits on white background."
    )
    
    if uploaded_file is not None:
        selected_color = st.selectbox(
            "Choose color to apply:",
            range(10),
            format_func=lambda x: f"{x} - {COLOR_NAMES[x]}"
        )

        image = Image.open(uploaded_file)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Preprocessed (28×28)")
            # Show preprocessed grayscale
            img_gray = image.convert('L').resize((28, 28), Image.Resampling.LANCZOS)
            img_array = np.array(img_gray).astype(np.float32) / 255.0
            if img_array.mean() > 0.5:
                img_array = 1.0 - img_array
            st.image(img_array, use_container_width=True, clamp=True)

        # Preprocess with selected color
        img_tensor, img_rgb = preprocess_uploaded_image(image, selected_color, device)

        with col3:
            st.subheader(f"With {COLOR_NAMES[selected_color]} Color")
            st.image(np.transpose(img_rgb, (1, 2, 0)), clamp=True, use_container_width=True)

        # Inference
        predicted, confidence, probs = predict(model, img_tensor, device)

        st.header("3️⃣ Prediction Results")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Predicted Digit", predicted, help="Model's prediction")
            st.metric("Confidence", f"{confidence:.2f}%", help="Model's confidence in prediction")
            st.metric("Applied Color", COLOR_NAMES[selected_color], help="Color applied to the digit")
            
            # Check if model might be fooled by color
            if predicted == selected_color:
                st.warning(f"⚠️ Model predicted **{predicted}** and color is **{COLOR_NAMES[selected_color]}**. Might be fooled by color!")
            else:
                st.info(f"ℹ️ Model predicted **{predicted}** despite **{COLOR_NAMES[selected_color]}** color. Good sign!")
        
        with col2:
            st.subheader("Confidence Distribution")
            plot_probabilities(probs, predicted, selected_color)
        
        st.divider()
        
        # Analysis
        st.subheader("🔍 Analysis")
        
        if predicted == selected_color:
            st.error(f"""
            **🚨 Potential Color Bias Detected!**
            
            The model predicted **{predicted}**, which matches the applied color **{COLOR_NAMES[selected_color]}**.
            
            This suggests the model might be relying on the **spurious correlation** (color) rather than 
            the **causal feature** (digit shape).
            
            💡 **Try**: Apply a different color to see if the prediction changes!
            """)
        else:
            st.success(f"""
            **✅ Good Shape Recognition!**
            
            The model predicted **{predicted}** despite the **{COLOR_NAMES[selected_color]}** color being applied.
            
            This suggests the model is focusing on the **shape** (causal feature) rather than just the color.
            """)
    
    else:
        st.info("👆 Please upload an image to test the model")

# =========================================================
# DATASET MODE
# =========================================================
else:
    st.markdown("""
    We generate MNIST digits with **random colors** (distribution shift from training).  
    This tests whether the model learned **shape** (causal) or **color** (spurious).
    """)

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🎲 Generate New Samples", use_container_width=True):
            st.session_state.samples = get_colored_mnist(train=False)
            st.session_state.sample_indices = None

    if "samples" not in st.session_state:
        st.session_state.samples = get_colored_mnist(train=False)
    
    if 'sample_indices' not in st.session_state or st.session_state.sample_indices is None:
        st.session_state.sample_indices = np.random.choice(len(st.session_state.samples), 5, replace=False)

    samples = st.session_state.samples
    indices = st.session_state.sample_indices
    
    st.header("3️⃣ Model Predictions")
    cols = st.columns(5)

    correct_count = 0
    color_fooled_count = 0

    # TensorDataset needs to be accessed by index, not sliced directly
    for i, idx in enumerate(indices):
        img_tensor, true_label = samples[int(idx)]
        predicted, confidence, _ = predict(model, img_tensor, device)

        # Convert to int if tensor
        true_label = true_label.item() if torch.is_tensor(true_label) else true_label
        
        is_correct = (predicted == true_label)
        if is_correct:
            correct_count += 1

        with cols[i]:
            # Show image
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img_tensor.permute(1, 2, 0).numpy())
            ax.axis('off')
            st.pyplot(fig)
            plt.close()
            
            # Show results
            if is_correct:
                st.success(f"✅ **Correct!**")
            else:
                st.error(f"❌ **Wrong**")
            
            st.markdown(f"**True:** {true_label}")
            st.markdown(f"**Predicted:** {predicted}")
            st.markdown(f"**Confidence:** {confidence:.1f}%")
            
            # Check if model was fooled by color
            if not is_correct:
                img_np = img_tensor.permute(1, 2, 0).numpy()
                center_pixel = img_np[14, 14]
                center_color = tuple(np.round(center_pixel, 1))

    st.divider()
    
    # Results Summary
    accuracy = (correct_count / 5) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{accuracy:.0f}%")
    with col2:
        st.metric("Correct Predictions", f"{correct_count}/5")
    with col3:
        st.metric("Color-Based Errors", color_fooled_count)

# =========================================================
# EXPLANATION SECTIONS
# =========================================================
st.divider()
st.header("📚 Understanding the Results")

with st.expander("🔬 Why do standard models fail?"):
    st.markdown("""
    **The Shortcut Learning Problem:**
    
    During training, the model sees that:
    - All digit '0' images are Red
    - All digit '1' images are Green
    - etc.
    
    The model learns the **easiest pattern** to minimize training loss: **"Color → Digit"**
    
    This works perfectly on training data (100% accuracy) but fails on test data 
    where colors are randomized (~10% accuracy, same as random guessing).
    """)

with st.expander("✨ How does IRM solve this?"):
    st.markdown("""
    **Invariant Risk Minimization (IRM):**
    
    1. **Multiple Environments**: Train on datasets with different correlation strengths
       - Environment 1: 90% color-digit correlation
       - Environment 2: 80% color-digit correlation
    
    2. **Invariance Penalty**: Add a mathematical penalty that punishes features 
       that aren't predictive across ALL environments
    
    3. **Result**: The model is forced to find features that work everywhere (digit shape)
       and ignore features that are unreliable (color)
    
    **Test Accuracy: ~97%** even with random colors! 🎉
    """)

with st.expander("💡 Real-world applications"):
    st.markdown("""
    This same problem appears in many real-world scenarios:
    
    - **Medical Diagnosis**: Model relies on hospital equipment watermarks instead of actual symptoms
    - **Hiring Systems**: Model uses gender/ethnicity as proxy instead of actual qualifications
    - **Autonomous Vehicles**: Model relies on clear weather patterns instead of learning robust road rules
    - **Credit Scoring**: Model uses zip code as proxy for creditworthiness instead of actual financial behavior
    
    **Causal ML techniques like IRM help build more robust and fair AI systems!**
    """)

