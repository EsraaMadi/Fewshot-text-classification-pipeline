from fastapi import FastAPI
from setfit import SetFitModel, SetFitTrainer
from text_categorizer.build_model import BuildModel
import yaml
import uvicorn
import pandas as pd
import os

#general set up 
config_path = "../config.yml"
data_name = 'Restaurant_Reviews'

# load configration
ymlfile = open(config_path)
cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
labels = cfg[data_name]['supervised-model-label_map']
labels = {value: key for key, value in labels.items()}
print(labels)

# load model
model = SetFitModel.from_pretrained(cfg[data_name]['supervised-model-path'])

app = FastAPI()

def process_text_en(text):
    data = pd.DataFrame({'new_text': [text]})
    preprocess_obj = BuildModel(use_pretrained_model=False, data=data,
                                col_name='new_text')
    preprocess_obj.clean_data(delete_null_flag=True,
                              split_flag=False,
                              remove_special_chart=False,
                              ignore_short_flag=True)   
    df = getattr(preprocess_obj, 'data_lang')['en']
    return list(df['new_text'].values)
    
@app.get('/')
async def index():
    return {"Message": "This is Index"}


@app.get('/preprocess/{full_text:str}')
async def preprocess_en(full_text: str):
    return {"processed_text": process_text_en(full_text)}

@app.get('/predict/{full_text:str}')
async def predict(full_text: str):
    full_text = full_text.replace('+', ' ')
    text_lst = process_text_en(full_text)
    pred_labels_id = model(text_lst)
    print(text_lst, pred_labels_id)
    pred_labels = list(map(lambda x: labels[x], pred_labels_id.tolist()))
    pred_labels_prob = [round(prob[l_id].item(), 2) for prob, l_id in zip(model.predict_proba(text_lst), pred_labels_id)]
    return {t: [l,p] for t, l, p in zip(text_lst, pred_labels, pred_labels_prob)}