import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.llms import OpenAI
from ingest_data import embed_doc
from query_data import _template, CONDENSE_QUESTION_PROMPT, QA_PROMPT, get_chain
import pickle
import os

os.environ["OPENAI_API_KEY"] = "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"

st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
import streamlit as st

footer="""<style>

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: right;
}
</style>
<div class="footer">
<p>Made with ❤ and \U0001F916 by Mobilefirst</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)

with open("style.css") as f:
    print("loaded")
    st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

st.header("Chat with Pdf Demo")

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = ""

if "past" not in st.session_state:
    st.session_state["past"] = []

if "generated" not in st.session_state:
    st.session_state["generated"] = []
    
def process_file(uploaded_file):
    if uploaded_file is not None:
        print("yes")
    else:
        print("no")
    print("here")
    with open(uploaded_file.name,"wb") as f:
        f.write(uploaded_file.getbuffer())
        print(uploaded_file.name)
        st.write("File Uploaded successfully")
        
        with st.spinner("Document is being vectorized...."):
            embed_doc(uploaded_file.name)
            


uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None and st.session_state.uploaded_file_name != uploaded_file.name:
    print("okay yes")
    print(type(st.session_state.uploaded_file_name))
    print(st.session_state.uploaded_file_name)
    st.session_state.uploaded_file_name = uploaded_file.name
    print("then")
    print(st.session_state.uploaded_file_name)
    process_file(uploaded_file)     

if "vectorstore.pkl" in os.listdir("."):
    with open("vectorstore.pkl","rb") as f:
        print("hello")
        vectorstore=pickle.load(f)
        if uploaded_file is not None and uploaded_file.name in os.listdir("."):
            os.remove(uploaded_file.name)
        
        print("Loading vectorscore...")
    chain=get_chain(vectorstore)
else:
    print("Bye")


def get_text():
    input_text = st.text_input("You: ", value="", key="input")
    return input_text


user_input = get_text()
print(st.session_state.input)
print(user_input)

if user_input:
    
    docs=vectorstore.similarity_search(user_input)

    print(len(docs))
    output = chain.run(input=user_input, vectorstore=vectorstore, context=docs[:2], chat_history=[], question=user_input, QA_PROMPT=QA_PROMPT, CONDENSE_QUESTION_PROMPT=CONDENSE_QUESTION_PROMPT, template=_template)

    st.session_state.past.append(user_input)
    print(st.session_state.past)
    st.session_state.generated.append(output)
    print(st.session_state.past)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")