import pickle
import streamlit as st 


pickle_in = open("BankNote.pkl","rb")
classifier=pickle.load(pickle_in)


def predict_note_authentication(Variance,Skewness,Curtosis,Entropy):
    prediction=classifier.predict([[Variance,Skewness,Curtosis,Entropy]])
    print(prediction)
    return prediction

def main():
    st.title("Bank Note Authenticate Prediction")
    Variance = st.text_input("Variance")
    Skewness = st.text_input("Skewness")
    Curtosis = st.text_input("Curtosis")
    Entropy = st.text_input("Entropy")
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(Variance,Skewness,Curtosis,Entropy)
    st.success('The output is {}'.format(result))
    

if __name__=='__main__':
    main()