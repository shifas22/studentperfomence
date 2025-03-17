import streamlit as st
import pandas as pd
import joblib
model=joblib.load('Model.pkl')
scaler=joblib.load('Scaler.pkl')
encoder=joblib.load('Encoder.pkl')


#st.title('Student perfromence perfomance')
#st.header('Shifas')
#st.subheader('Salman')
# st.write('enter your name')
# st.text_input('Enter your Age :')
# a=st.number_input('enter a number')
# st.selectbox('Choose',('Yes','No'))
# st.radio('Gender',options=['male','female'])
# b=st.button('Submit')


# if b:
#     st.write(a)

# num1=st.number_input('Enter the First Number')
# num2=st.number_input('Enter the Second Number')

# a=st.selectbox('Choose',('Add','Sub','Mult','Div'))

# if a =='Add':
#     if st.button('Submit'):
#         st.success(num1+num2)
# elif a == 'Sub':
#     if st.button('Submit'):
#         st.success(num1-num2)
# elif a =='Mult':
#     if st. button('Submit'):  
#         st.success(num1*num2)
# elif a =='Div' :
#     if st.button('Submit'):
#         if num2 == 0:
#             st.error('You cant give denominator as 0')
#         else :
#             st.success(num1/num2)


  ####################################################

  #students prfmnc prediction
  
hour_studied= st.number_input('Hours Studied')
prev_score=st.number_input('Previous Score')
sleep_hours=st.number_input('Sleep Hours')
peper=st.number_input('Sample Qestion paper practiced')
eca=st.selectbox('Extracurricular Activity ',('Yes','No'))
eca=encoder.transform([eca])
# st.write(eca)

data=pd.DataFrame({"Hours Studied" :hour_studied,"Previous Scores" :prev_score,"Sleep Hours" : sleep_hours,"Sample Question Papers Practiced":peper,"ECA":eca})
# st.write(data)

scaled_data=scaler.transform(data)

# st.write(scaled_data)

if st.button('predict'):
    prediction=model.predict(scaled_data)[0]
    st.write(prediction)


