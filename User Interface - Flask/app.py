from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import random
from tensorflow import keras
from keras.layers import Input
from keras.models import Model, load_model

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    result = ''
    extra = ''
    answer = {}

    if(request.method == 'POST'):
        exit_commands = ("sampai jumpa", "berhenti", "selesai", "keluar", "goodbye", "bye", "stop", "exit")
        user_input = request.form['user_input']

        for exit_command in exit_commands:
            if exit_command in user_input:
                result = "Baik, semoga jawabanku memuaskan. Sampai jumpa lagi :D"
                answer = {'result': result}
                return jsonify(answer)
        
        result = chatbot_answer(user_input)

        feedback = ['Ada yang ingin ditanyakan lagi?', 
                    'Ada pertanyaan lain?', 
                    'Tanya lagi dong!', 
                    'Beri aku pertanyaan lagi!', 
                    'Yuk bertanya lagi!', 
                    'Coba pertanyaan yang lain!']
        extra = random.choice(feedback)
        answer = {'result': result, 'extra': extra}

        return jsonify(answer)

def chatbot_answer(user_input):
    chatbot = ChatBot()
    return chatbot.generate_response(user_input)

def load_our_model():
    training_model = load_model('training_model.h5')
    encoder_inputs = training_model.input[0] # Mengambil encoder input
    _, state_h_enc, state_c_enc = training_model.layers[4].output # Mengambil output dari LSTM encoder
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = Input(shape=(maxlen_answers,))
    decoder_embedding = training_model.layers[3]
    decoder_lstm = training_model.layers[5]
    decoder_dense = training_model.layers[6]

    latent_dim = 256
    decoder_state_input_hidden = Input(shape=(latent_dim,))
    decoder_state_input_cell = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
    decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_embedding(decoder_inputs), initial_state=decoder_states_inputs)
    decoder_states = [state_hidden, state_cell]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

def decode_response(input_matrix):
    states_value = encoder_model.predict(input_matrix)
    target_seq = np.zeros((1 , maxlen_answers))
    target_seq[0, 0] = tokenizer.word_index['start']
    
    decoded_sentence = ''
    stop_condition = False

    while not stop_condition:
      output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
      sampled_token_index = np.argmax(output_tokens[0, -1, :]) 
      sampled_token = None

      for word, index in tokenizer.word_index.items():
          if sampled_token_index == index:
              decoded_sentence += f' {word}'
              sampled_token = word

      if (sampled_token == 'end' or len(decoded_sentence.split()) > maxlen_answers):
          stop_condition = True

      target_seq = np.zeros((1, maxlen_answers))  
      target_seq[0, 0] = sampled_token_index
      states_value = [hidden_state, cell_state]

    return decoded_sentence

class ChatBot:    
    def string_to_matrix(self, user_input):
        tokens_list = tokenizer.texts_to_sequences([user_input])

        return keras.preprocessing.sequence.pad_sequences(tokens_list, maxlen=maxlen_questions, padding='post')
  
    def generate_response(self, user_input):
        input_matrix = self.string_to_matrix(user_input)
        chatbot_response = decode_response(input_matrix)
        chatbot_response = chatbot_response.replace("end",'')
        
        return chatbot_response
    
if(__name__ == '__main__'):
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open('maxlen_questions.pkl', 'rb') as f:
        maxlen_questions = pickle.load(f)
    
    with open('maxlen_answers.pkl', 'rb') as f:
        maxlen_answers = pickle.load(f)
    
    encoder_model, decoder_model = load_our_model()
    app.run()