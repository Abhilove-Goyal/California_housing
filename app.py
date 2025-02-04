import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Title
st.title('My Streamlit App')

# Slider for selecting a number
slider_value = st.slider('Select a number:', 0, 100, 50)
st.write('Hehe Hiii! pls tell me about yourself: ')

# Text input for the user's name
user_input = st.text_input('Enter your name: ')
st.write(f"Hello Master {user_input}")

# padhle bhai 

st.title("Create static table using st.table(df)")
df=pd.DataFrame({
    'first column' : [1,2,3,4],
    'Second column':[5,6,7,8]
})
st.table(df)

st.title("This time attempting to create data frame without df")
if st.checkbox('Show dataframe'):
    st.write(pd.DataFrame({
        'first' : [1,2,3,4],
        'second' : [5,6,7,8]
    })
)

dataframe=pd.DataFrame(
    np.random.randn(10,20),
    columns=('col %d' %i for i in range (20))
)
st.dataframe(dataframe.style.highlight_max(axis=1))

#making graph
chart_data=pd.DataFrame(
    np.random.randn(10,3),
    columns=['a','b','c']
)
st.line_chart(chart_data)

#PLOTTING A Map 
map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'])

st.map(map_data)


x = st.slider('x')  # 👈 this is a widget
st.write(x, 'squared is', x * x)

st.text_input("Your name", key="name")

# You can access the value at any point with:
st.session_state.name
















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
