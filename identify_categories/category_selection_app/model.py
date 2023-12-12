from bertopic import BERTopic
import pickle
import streamlit as st
import numba
import math
import _io
import yaml
import time

class BertStreamlit:
 
    def __init__(self,
                 model_file,
                 parameter_file):
        # model
        self.topic_model = BERTopic.load(model_file)
        #model parameters
        self.file = open(parameter_file,'rb')
        self.embeddings = pickle.load(self.file)
        self.probs = pickle.load(self.file)
        self.topics = pickle.load(self.file)
        self.data_lang = pickle.load(self.file)
        self.topics_labels = list(self.topic_model.get_topic_info()['CustomName'])
        self.file.close()
    
    #-------------------------------------------------------------
    
    def get_top_words(self, topic_id):
        # return bar charts out of the c-TF-IDF scores for each topic representation
        fig = self.topic_model.visualize_barchart([topic_id])
        return fig

    #-------------------------------------------------------------
    
    @st.cache(hash_funcs={dict: lambda _: None}, ttl=3600)
    def get_hierarchical_topics_chart(self,
                                      txt_lst,
                                      load_save_fig=True,
                                      fig_file=''):
        # The topics that were created can be hierarchically reduced. In order to understand the potential hierarchical
        if load_save_fig:
            file = open(fig_file,'rb')
            fig = pickle.load(file)
        else:
            # structure of the topics
            hierarchical_topics = self.topic_model.hierarchical_topics(txt_lst,
                                                                        self.topics)
            fig = self.topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics,
                                                       custom_labels=self.topics_labels)
        return fig
    
    #-------------------------------------------------------------
    
    @st.cache(hash_funcs={dict: lambda _: None}, ttl=3600)
    def get_visualize_documents_chart(self,
                                      txt_lst):
        fig = self.topic_model.visualize_documents(txt_lst,
#                                                    topics=[selected_topics_ids],
                                                   embeddings=self.embeddings,
                                                   custom_labels=self.topics_labels)
        return fig
    
    #-------------------------------------------------------------
    
    def save_topic(self,
                   config_path,
                   topic_id,
                   topic_name,
                   dataset_name):
        cfg = {}
        with open(config_path, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        with open(config_path, 'w') as ymlfile:
            if cfg[dataset_name].get('target_topics', 0) == 0:
                cfg[dataset_name]['target_topics'] = {}
            cfg[dataset_name]['target_topics'][topic_id] = topic_name 
            yaml.dump(cfg, ymlfile)
            return "Topic/Category Configuration is saved!"
    
    #-------------------------------------------------------------
        
    def get_saved_topic(self,
                        config_path,
                        dataset_name):
        with open(config_path, 'r') as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            return cfg[dataset_name].get('target_topics', 0)