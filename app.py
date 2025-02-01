import streamlit as st
import matplotlib.pyplot as plt
import numpy as np



st.title('My streamlit app')
slider_value = st.slider('Select a number:', 0, 100, 50)
st.write('Hehe Hiii! pls tell me about yourself: ')
user_input=st.text_input('Enter your name: ')
st.write(f"Hello Master {user_input}")

x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
plt.plot(x, y)

# Display the plot in Streamlit
st.pyplot(plt)