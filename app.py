from flask import Flask, render_template, request, session
import openai
import os
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

app = Flask(__name__)
app.secret_key = 'supersecretkey'

messages = []
# openai.api_key = ''
os.environ['OPENAI_API_KEY'] = '' 

def initialize_retriever():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    return vectorstore.as_retriever()

def initialize_llm():
    return ChatOpenAI(model_name='gpt-4o', temperature=0)

def initialize_memory():
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)

retriever = initialize_retriever()
llm = initialize_llm()
memory = initialize_memory()

def get_response(user_input):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory
    )
    response = qa_chain.invoke({'query': user_input})
    if isinstance(response, dict) and 'result' in response:
        return response['result']
    else:
        return str(response)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

def ExplainMore():
    return get_response("Please explain more in detail. Answer in English.")

def OriginalCase():
    return get_response("Please tell me: What is the original case for the current discussion? Answer in English.")

def SimilarScenario():
    return get_response("Please generate a similar scenario based on our current discussion for me to practice. Answer in English.")

def RelevantTheories():
    return get_response("Please tell me: What data and theories form the basis of our current discussion, and provide the sources of this information. Answer in English.")

@app.route('/main', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        input_text = request.form['input_text']
        previous_input = session.get('previous_system_response', '')
        
        session['previous_system_response'] = input_text
        
        if 'Explain More' in input_text:
            result_text = ExplainMore()
        elif 'Original Case' in input_text:
            result_text = OriginalCase()
        elif 'Similar Scenario' in input_text:
            result_text = SimilarScenario()
        elif 'Relevant Theories' in input_text:
            result_text = RelevantTheories()
        else:
            prompt_template = f"Answer in English. Based on my input, please determine my intent. If my input doesn't mention 'generate a new case' or similar terms, it means I am providing feedback on an existing case. You should compare my feedback with previous data handling methods, then generate suggestions and data sources. If I mention 'generate a new case,' then please generate a new case description, and after finishing the description, ask: Based on this case description, what will your next decision and action be?\n\n You don't need to explain your reasoning, just start outputting the relevant content. If it's Intent 1, this is the last system message (case description): {previous_input}. If it's Intent 2, please ignore the last system message and generate a new case description.\n\n This is my input:"
            result_text = get_response(prompt_template + input_text)
        
        session['previous_system_response'] = result_text
        
        messages.append({'text': input_text, 'type': 'user-message'})
        messages.append({'text': result_text, 'type': 'bot-message'})
        return render_template('main.html', messages=messages)

    return render_template('main.html', messages=messages)

@app.route('/data')
def data():
    return render_template('data.html')

if __name__ == '__main__':
    app.run(debug=True)
