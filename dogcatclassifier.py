import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('dogcat.h5')
  return model
model=load_model()
st.write("""
# Cat and Dog Classifier"""
)
file=st.file_uploader("Choose a photo of dog or cat",type=["jpg","png"])
import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(100,100)
    image=ImageOps.fit(image_data,size,Image.LANCZOS)
    img=np.asarray(image)
    img_reshape=img[np.newaxis,...]
    prediction=model.predict(img_reshape)
    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Cat', 'Dog']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)
  
  # User feedback section
    st.header("User Feedback")
    user_feedback = st.checkbox("Was this classification correct?")

    if st.button("Submit Feedback"):
        collect_user_feedback(file.name, class_names[np.argmax(prediction)], user_feedback)
        st.success("Feedback submitted successfully!")

    # Display statistics or insights based on user interactions and collected feedback
    if st.session_state.feedback_data_exists:
        feedback_data = pd.read_csv('user_feedback.csv')

        total_feedback = len(feedback_data)
        correct_feedback = len(feedback_data[feedback_data['UserFeedback']])

        st.subheader("Feedback Statistics")
        st.write(f"Total Feedback Submitted: {total_feedback}")
        st.write(f"Correct Feedback: {correct_feedback}")
        st.write(f"Feedback Accuracy: {correct_feedback / total_feedback:.2%}")
