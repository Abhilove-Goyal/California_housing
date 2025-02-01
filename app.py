import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Title
st.title('My Streamlit App')

# Slider for selecting a number
slider_value = st.slider('Select a number:', 0, 100, 50)
st.write('Hehe Hiii! pls tell me about yourself: ')

# Text input for the user's name
user_input = st.text_input('Enter your name: ')
st.write(f"Hello Master {user_input}")

# Embed Power BI Report using iframe (via markdown)
st.markdown(
    '''
        <iframe title="AddidasSalesAnalysis" width="1140" height="541.25" src="https://app.powerbi.com/reportEmbed?reportId=5d13ab04-8fa3-43d2-9aeb-72f7d24387de&autoAuth=true&embeddedDemo=true" frameborder="0" allowFullScreen="true"></iframe>
    ''', unsafe_allow_html=True)

# Plot using matplotlib
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a plot
plt.plot(x, y)

# Display the plot in Streamlit
st.pyplot(plt)
