import pandas as pd
import re
import yaml
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from transformers import pipeline
import pickle
import os


class BuildModel:
    
    """
    A class used to train bert model (Bertopic) to use in topic
    modeling tasks and categorizing text

    ...

    Attributes
    ----------
    data : dataframe
        a dataframe that has a string column and other related
        columns
    col_name : str
        the name of the column that has the text

    Methods
    -------
    clean_data(delete_null_flag=True,
                split_flag=True,
                ignore_short_flag=True)
        clean the text that would be the input for topic model
    
    train_model(lang='en')
        train bert topic to obtain topics from givin text
        
    save_model(model_path)
        save trained model in a giving path
        
    topics_names_process(self)
        phrasing topic names in more readable way
    """

    def __init__(self,
                 use_pretrained_model=False, 
                 **kwargs):
        """
        Parameters
        ----------
        data : dataframe
            a dataframe that has a string column and other related
            columns
        col_name : str
            the name of the column that has the text
        """
        # train new model
        if not use_pretrained_model:
            self.data = kwargs['data']
            self.text_col = kwargs['col_name']
        else:
            self.topic_model = BERTopic.load(f"{kwargs['model_path']}/model_01")
            file = open(f"{kwargs['model_path']}/model_parameters.pkl",'rb')
            self.embeddings = pickle.load(file)
            self.probs = pickle.load(file)
            self.topics = pickle.load(file)
            self.data_lang = pickle.load(file)
            
        
    def clean_data(self,
                   delete_null_flag=True,
                   split_flag=True,
                   remove_special_chart=True,
                   ignore_short_flag=True):
        """clean the text that would be the input for topic model

        If arguments aren't passed in, the default arguments are used.

        Parameters
        ----------
        delete_null_flag : bool, optional
            Flag to delete all rows that have missing in the traget text column 
            (default is True)
            
        split_flag : bool, optional
            Flag to split long text by defined sperators (default is True)
            
        ignore_short_flag : bool, optional
            Flag to delete comments that has less than 4 letters (default is True)
        """
        print('Data Cleaning for:',self.data.shape[0], 'rows')
        # 1- delete all rows that has null values in the target column (text column)
        if delete_null_flag:
            self.data = self.data[~self.data[self.text_col].isnull()].reset_index(drop=True)
            print('Data Cleaning - remove missing text:',self.data.shape[0], 'rows')
            
            # fill missing values if needed
            self.data.fillna('Unknown')
        
        # 2- split long text that probably talk about more than one topic to different cells 
        # we define long text if it has these sperators (numerical list, dash, dot)
        if split_flag:
            self.data[self.text_col] =self.data[self.text_col].map(lambda txt: re.sub(r'(\d+\s*-\s*)', '.', str(txt)))
            self.data[self.text_col] =self.data[self.text_col].map(lambda txt: re.sub(r'(\s+-\s+)', '.', str(txt)))
            self.data[self.text_col] =self.data[self.text_col].map(lambda txt: re.sub(r'(\s*,\s*)', '.', str(txt)))
            self.data[self.text_col] =self.data[self.text_col].map(lambda txt: txt.split('.'))
            self.data =self.data.explode(self.text_col)
            print('Data Cleaning - split text:',self.data.shape[0], 'rows')
        
        if remove_special_chart:
        # 3- remove special charachters from the text
            self.data[self.text_col] =self.data[self.text_col].map(lambda txt: re.sub(r'[-?|$||!_— .…/]', ' ', txt))
            self.data[self.text_col] =self.data[self.text_col].map(lambda txt: txt.replace('\n', '.'))
            print('Data Cleaning - remove special charachters from the text:',self.data.shape[0], 'rows')
        
        # 4- remove white spaces
        self.data[self.text_col] =self.data[self.text_col].map(lambda txt: ' '.join(txt.split()))
        print('Data Cleaning - remove white spaces:',self.data.shape[0], 'rows')
        
        # 5- delete comment less than 4 charchters
        if ignore_short_flag:
            self.data =self.data[self.data[self.text_col].map(lambda txt: True if len(txt) > 3 else False)]
            print('Data Cleaning - ignore short text(less than 4 charchters):',self.data.shape[0], 'rows')
            
        # 6- get language and split dataset
        self.data_lang = {}
        self.data_lang['ar'] =self.data[self.data[self.text_col].map(lambda txt: bool(re.compile(r"[\u0600-\u06FF]+").search(txt)))]
        print('Data Cleaning - Arabic text:',self.data_lang['ar'].shape[0])
        self.data_lang['en'] =self.data[self.data[self.text_col].map(lambda txt: bool(re.compile(r"[A-Za-z]+").search(txt)))]
        print('Data Cleaning - English text:',self.data_lang['en'].shape[0])
        
        
    def train_model(self,
                    lang='en'):
        """ train bert topic to obtain topics from givin text

        If arguments aren't passed in, the default arguments are used.

        Parameters
        ----------
        lang : string, optional
            language of the trained data (default is 'en')
            

        Returns
        ------
        topic_model: bertmodel
            trained model in giving text
        
        topics: list
            list contains topic id for each givin text
            
        probs: list
            list contains the probablity ofeach text to belong to obtained topic
            
        embeddings: list of list
            list of transformed text to vectors
            
        text_lst: list
            list of training data (text)
        """
            
        # get training data
        text_lst = self.data_lang[lang][self.text_col].values
            
        # build model
        self.topic_model = BERTopic(nr_topics='auto',
                                    embedding_model='all-MiniLM-L6-v2' )
        # train and get topics for giving text
        self.topics, self.probs = self.topic_model.fit_transform(text_lst)
        sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = sentence_model.encode(text_lst, show_progress_bar=False)
        
        # preprocess topics names
        self.topics_names_process()
        
        # update input data with model output
        self.data_lang[lang]['topic_id'] = self.topics
        self.data_lang[lang]['topic_prob'] = self.probs
        topics_map = dict(self.topic_model.get_topic_info()[['Topic', 'CustomName']].values)
        self.data_lang[lang]['topic_name'] = self.data_lang[lang]['topic_id'].map(topics_map)
        
        return self.topic_model, self.topics, self.probs, self.embeddings, text_lst, self.data_lang
    

    def save_model(self, 
                   model_path,
                   config_file_path,
                   config_sec):
        """ save trained model in a giving path

        If arguments aren't passed in, the default arguments are used.

        Parameters
        ----------         
        model_path : str
            path to save th train model on to use it later

        config_sec : str
            name of data section in yaml configration file
        """
        # check model path
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        
        # save model
        self.topic_model.save(model_path+'/model_01')
        
        # save model paramters
        file = open(f'{model_path}/model_parameters.pkl','wb')
        pickle.dump(self.embeddings, file)
        pickle.dump(self.probs, file)
        pickle.dump(self.topics, file)
        pickle.dump(self.data_lang, file)

        # save model path in the configration file
        cfg = {}
        with open(config_file_path) as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            cfg[config_sec]['unsupervised-model-path'] = model_path

        with open(config_file_path, 'w') as ymlfile:
            yaml.dump(cfg, ymlfile)

    def topics_names_process(self):
        """ phrasing topic names in more readable way
        """
        topics_name_lst = [i.split('_') for i in self.topic_model.get_topic_info()['Name']]
        topics_name_lst = [f'Topic {i[0]} ---> '+ ', '.join(i[1:]) for i in topics_name_lst]
        self.topic_model.set_topic_labels(topics_name_lst)
        
    def set_data(self,
                 df,
                 lang='en'):
        self.data_lang[lang] = df
        
    def get_trained_model(self):
        return self.topic_model, self.topics, self.probs, self.embeddings, self.data_lang
        