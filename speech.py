import speech_recognition as spr
import soundfile
from termcolor import colored
import os
from transformers import AutoModelForCausalLM,AutoTokenizer
import torch 

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
recog = spr.Recognizer()

mc = spr.Microphone(device_index=1)
print('listening : ')
step =0
while True:
    txt = ''
    with mc as source:  
        recog.adjust_for_ambient_noise(source)
        audio = recog.listen(source)
        txt_ = recog.recognize_google(audio,language = 'en-US')
        if txt_ == 'stop listening':
            break
        txt = colored(txt_,'green',attrs=['bold'])
        print(txt)
        new_user_input_ids = tokenizer.encode(txt_ + tokenizer.eos_token,return_tensors='pt')
        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids,new_user_input_ids],dim=1) if step >0 else new_user_input_ids
        # generated a response while limiting the total chat history to 1000 tokens
        chat_history_ids = model.generate(bot_input_ids,max_length = 1000,pad_token_id = tokenizer.eos_token_id)
        aws = 'agnet : {}' . format(tokenizer.decode(chat_history_ids[:,bot_input_ids.shape[-1]:][0],skip_special_tokens=True))
        aws = colored(aws,'cyan',attrs=['bold'])
        print(aws)
        step += 1
            
        

    
