import matplotlib.pyplot as plt
import streamlit as st

def plot_probabilities(probs, predicted_digit, selected_color):
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(10), probs * 100)
    bars[predicted_digit].set_color('green')
    bars[selected_color].set_color('orange')

    ax.set_xlabel('Digit')
    ax.set_ylabel('Confidence (%)')
    ax.set_xticks(range(10))
    ax.set_ylim([0, 100])
    st.pyplot(fig)
    plt.close()
