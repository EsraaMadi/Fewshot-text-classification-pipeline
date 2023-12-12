from text_categorizer.build_model import BuildModel
from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss
import argilla as rg
import pandas as pd
from datetime import date
from sklearn.metrics import confusion_matrix
import yaml
import pickle
import os
import numpy as np
import datasets as dt


class BuildFewShotModel:
    
    def __init__(self,
                 config_file,
                 data_file_name,
                 business_kpi,
                 active_learning_method=0):
        
        print('=============================================')
        print('===============     Steps     ===============')
        print('=============================================')
        # load configration file
        self.config_file = config_file
        self.data_file_name = data_file_name
        with open(config_file) as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
            self.cfg_data = self.cfg[self.data_file_name]
        print("Step 0: Loading configration file --> Done")
        
        # start from unsupervised result
        self.first_label_flag = True
        
        # get labels
        self.labels = list(set(self.cfg_data['target_topics'].values())) #+ ['Others']
        
        # score matrix
        self.score_path = f'../models/{self.data_file_name}/Supervised_model/scores.pkl'
        self.test_file_path = f'../models/{self.data_file_name}/Supervised_model/test.pkl'
        self.scores = pd.DataFrame(columns=['id', 'train_set', 'test_set', 'score', 'creation_date']+self.labels)
        print("Step 0: Loading score matrix --> Done")
        
        # check if there is a model trained before
        if self.has_trained_model():
            self.first_label_flag = False
            self.model = SetFitModel.from_pretrained(self.sv_model_path)
            print("Step 0: Loading pretrained model --> Done")
            
            # load score matrix
            try:
                self.score_path = self.cfg_data['supervised-model-score-path']
                file = open(self.score_path,'rb')
                self.scores = pickle.load(file)
            except:
                pass
        
            
        
        
        # get argilla dataset name
        self.rg_dataset_name = self.cfg_data['rg_dataset_name']
        # in the future we need to setup workspace for each domain , currently it's an issue in the arrgilla https://github.com/argilla-io/argilla/issues/2402
        rg.init(api_key=business_kpi, api_url="http://localhost:6901")
        print(f"Step 0: setting argilla workspace {rg.get_workspace()}--> Done", )
        
        
        # set active learning method
        self.active_learning_ops = ['Overall accuracy', 'Per category accuracy', 'iterations']
        self.ac = self.active_learning_ops[active_learning_method]
        print(f"Step 0: Seting active learning: {self.ac}")
        
        
    # -----------------------------------------------------------------
    
    
    def has_trained_model(self):
        
            # get unsupervised model path
            self.usv_model_path = self.cfg_data['unsupervised-model-path']
            
            # check if train a model before for supervised classification
            if self.cfg_data.get('supervised-model-path', 0):
                self.sv_model_path = self.cfg_data['supervised-model-path']
                print("Step 0: Check if there is a pretrained model --> Yes")
                return True
            else:
                print("Step 0: Check if there is a pretrained model --> No")
                return False
        
        
    # -----------------------------------------------------------------
    
     
    def has_labeling_running(self):
        try:
            rg.load(self.rg_dataset_name).to_pandas()
            print("Step 1: Check if we have lunched a labeling process before --> yes")
            return (self.first_label_flag or not self.is_finish_labeling(step=1))
        except:
            print("Step 1: Check if we have lunched a labeling process before --> No")
            return False
        
        
    # -----------------------------------------------------------------
    
    
    def load_data_pool(self, lang='en'):
        
        # use unsupervised result
        if self.first_label_flag:
            #load pretrained model object
            model_obj = BuildModel(use_pretrained_model=True,
                                   model_path=self.usv_model_path)

            # load data
            self.data_pool = getattr(model_obj,'data_lang')[lang]
            print("Step 2: Loading data pool (original dataset) --> Done")
        
        # use previous labeld data from argilla
        else:
            # get recoreds that not labeled by user
            data_labeled_df = rg.load(self.rg_dataset_name).to_pandas()
            data_labeled_df['Topic_name_mapped'] = data_labeled_df['prediction'].map(lambda x: x[0][0])
            self.data_pool = data_labeled_df[data_labeled_df['status'] == 'Default']
            print("Step 2: Loading data pool (Unlabeled data from UI)--> Done")
        return self.data_pool
        
        
    # -----------------------------------------------------------------
    
       
    def get_metadata(self):
        ignore_columns = ['topic_name_mapped', 'topic_id', 'topic_prob', 'topic_name', 'prediction', 'prediction_agent']
        metadata_columns = [i for i in self.data_pool.columns if i not in ignore_columns]
        metadata_columns = metadata_columns if len(metadata_columns) <=50 else metadata_columns[0:49] # apply argilla limit
        metadata_df = self.data_pool[metadata_columns]
        # fill null numerical columns
        numeric_metadata_columns = metadata_df.select_dtypes(include=np.number).columns.tolist()
        metadata_df[numeric_metadata_columns] = metadata_df[numeric_metadata_columns].fillna(-1)
        # fill null text columns
        text_metadata_columns = [i for i in metadata_df.columns if i not in numeric_metadata_columns]
        metadata_df[text_metadata_columns] = metadata_df[text_metadata_columns].fillna('')
        return metadata_df.to_dict('records')
        
        
    # -----------------------------------------------------------------
    
      
    def prepare_data(self, row_limit=10):
        print("Step 3: Preparing data to fit UI template--> Done")
        # use unsupervised result
        if self.first_label_flag:
            
            # map required labels/topics
            self.data_pool['topic_name_mapped'] = self.data_pool[['topic_id']].apply(lambda x: self.cfg_data['target_topics'].get(x[0]),axis=1)
            
            # fill required column for loading process
            self.data_pool['prediction'] = self.data_pool[['topic_name_mapped','topic_prob']].apply(lambda x: [(x[0],x[1])] , axis=1)
            # self.data_pool['annotation'] = self.data_pool['Topic_name_mapped'].map(lambda x: [x])
            self.data_pool['prediction_agent'] = ['Unsupervised Algorithm'] * self.data_pool.shape[0]
            self.data_pool['id'] = list(range(self.data_pool.shape[0]))
            # fill metadate column
#             df.select_dtypes(include=np.number).columns.tolist()
#             ignore_columns = ['topic_name_mapped', 'topic_id', 'topic_prob', 'topic_name', 'prediction', 'prediction_agent']
#             metadata_columns = [i for i in self.data_pool.columns if i not in ignore_columns]
#             metadata_columns = metadata_columns if len(metadata_columns) <=50 else metadata_columns[0:50] # apply argilla limit
#             self.data_pool['metadata'] = self.data_pool[metadata_columns].to_dict('records')
            self.data_pool['metadata'] = self.get_metadata()
            self.data_pool.rename(columns={self.cfg_data['data info']['text_col']:'text'}, inplace=True)
            self.next_batch_data = self.data_pool
            
            
        # use previous labeld data from argilla
        else:
            print("Step 3: Get prediction for all unlabeled data")
            # get prediction for all data
            self.data_pool['sitfit_label_id'], self.data_pool['sitfit_label'], self.data_pool['sitfit_prob'] = self.predict_category(self.data_pool['text'].values)
            
            # get requsted number of rows with lowest prob
            self.next_batch_data = self.data_pool.nsmallest(row_limit, 'sitfit_prob')
            
            # fill required column for loading process
            self.next_batch_data.loc[:,'prediction_agent'] = ['supervised Algorithm'] *  self.next_batch_data.shape[0]
            # save new prediction for it
            self.next_batch_data['prediction'] = self.next_batch_data[['sitfit_label','sitfit_prob']].apply(lambda x: [(x[0],x[1])] , axis=1)
        
        self.next_batch_data = self.next_batch_data[['text','metadata',
                                                                   'prediction','id', 
                                                                   'prediction_agent']]
        return self.next_batch_data.shape
        
        
    # -----------------------------------------------------------------
    
    
    def upload_data_to_UI(self):
        # use unsupervised result
        if self.first_label_flag:
            try:
                rg.delete(name=self.rg_dataset_name)
#                 rg.copy("trainees-comments", name_of_copy=self.rg_dataset_name)
            except:
                pass
            
        # use previous labeld data from argilla
        else:
            # delete these records from argilla to add them again with diffrent values 
            rg.delete_records(name=self.rg_dataset_name, ids=self.next_batch_data['id'].values.tolist())
            
        # convert pandas dataframe to DatasetForTextClassification
        dataset_rg = rg.DatasetForTextClassification.from_pandas(self.next_batch_data)
        print(f"Step 4: Waiting upload {self.next_batch_data.shape[0]} records to UI ....")
        rg.log(dataset_rg,
               name=self.rg_dataset_name,
               tags = {"Business area": self.cfg_data['data info']["Business area"],
                       "Label type": self.cfg_data['data info']["Label type"],
                       "Year": self.cfg_data['data info']["Year"],
                       "Source":self.cfg_data['data info']["Source"]})
        print("Step 4: Uploading data to UI --> Done")
        print("Step 5: Waiting data to get labeled ....")
        
        
    # -----------------------------------------------------------------
    
     
    def is_finish_labeling(self, limit_row=25, step=5):
        
        # load data from UI
        data_labeled_df = rg.load(self.rg_dataset_name).to_pandas()
        
        # use unsupervised result
        if self.first_label_flag:
            annotated_labels_count = data_labeled_df['annotation'].value_counts()
            # print(annotated_labels_count.shape[0] ,len(self.labels) , annotated_labels_count.min() , limit_row)
            if annotated_labels_count.shape[0] >= len(self.labels) and annotated_labels_count.min() >= limit_row:
                print(f"Step {step}: Labeling data process in the UI --> Done")
                return True
        # use previous labeld data from argilla
        else:
            unlaneled_records = data_labeled_df[(data_labeled_df['status']=='Default') & (data_labeled_df['prediction_agent']=='supervised Algorithm')]
            if unlaneled_records.shape[0] == 0:
                print(f"Step {step}: Labeling data process in the UI --> Done")
                return True
#         print(f"Step {step}: Labeling data process in the UI --> Not Finished yet")
        return False
        
        
    # -----------------------------------------------------------------
    
    
    def load_arrgilla_annotated_data(self):
        all_data = rg.load(self.rg_dataset_name, query="status: Validated").to_pandas()
        all_data.to_excel('../../pydata_talk/data/labeled_data.xlsx')
        self.label_map = {i:self.labels.index(i) for i in all_data['annotation'].unique()}
        self.cfg[self.data_file_name]['supervised-model-label_map'] = self.label_map
        all_data['label'] = all_data['annotation'].map(self.label_map)
        return dt.Dataset.from_pandas(all_data[['text','label']])
        
        
    # -----------------------------------------------------------------
    
    
    
    def load_labeled_data(self):
        self.data_labeled_ds = self.load_arrgilla_annotated_data()
        # split data
        self.data_labeled_ds = self.data_labeled_ds.shuffle(seed=42)
        model_ul_data = self.data_labeled_ds.train_test_split(test_size=0.30)
        self.train_ul_ds = model_ul_data['train']
        self.test_ul_ds = model_ul_data['test']
        print(f"Step 6: Loading training data from UI, then split to training data {self.train_ul_ds.num_rows} rows, and testing data {self.test_ul_ds.num_rows} rows --> Done")
        return self.train_ul_ds.num_rows, self.test_ul_ds.num_rows
        
        
    # -----------------------------------------------------------------
    
    
    def train_model(self, transformer="sentence-transformers/paraphrase-mpnet-base-v2" ):
        print("Step 7: Train a new model ....")
        # Load new SetFit model from Hub
        model_ul = SetFitModel.from_pretrained(transformer)
        # Create trainer
        trainer_ul = SetFitTrainer(
            model=model_ul,
            train_dataset=self.train_ul_ds,
            eval_dataset=self.test_ul_ds,
            loss_class=CosineSimilarityLoss,
            batch_size=16,
            num_iterations=20, # The number of text pairs to generate
        )

        # Train and evaluate
        trainer_ul.train()
        self.last_score = round(trainer_ul.evaluate()['accuracy'],3)
        print("Step 7: Training new model --> Done")
        self.save_model(model_ul)
        self.save_test_example(model_ul)
        return(self.last_score)
       
        
    # -----------------------------------------------------------------
    
    
    def save_test_example(self, model_ul):
        file = open( self.test_file_path,'wb')
        pred_labels_id, pred_labels, pred_labels_prob = self.predict_category(self.test_ul_ds['text'])
        
        test_df = pd.DataFrame(data={'text': self.test_ul_ds['text'],
                                     'pred_labels_id':pred_labels_id,
                                     'pred_labels':pred_labels,
                                     'pred_labels_prob':pred_labels_prob,
                                     'actual_label_id': self.test_ul_ds['label'],
                                     'actual_label': list(map(lambda x: self.labels[x], self.test_ul_ds['label']))})
        pickle.dump(test_df, file)
        
    # -----------------------------------------------------------------
    
    
    def save_model(self, model_ul):
        
        if self.first_label_flag or self.scores[self.scores['score']<self.last_score].shape[0] > 0:
            
            # save model localy
            self.model = model_ul
            self.sv_model_path = f'../models/{self.data_file_name}/Supervised_model/scfhs_topic_classifier/'
            self.cfg[self.data_file_name]['supervised-model-path'] = self.sv_model_path
            self.cfg[self.data_file_name]['supervised-model-score-path'] = self.score_path
            self.first_label_flag = False
            
            # check model path
            if not os.path.exists(self.sv_model_path):
                os.makedirs(self.sv_model_path)
            
            self.model.save_pretrained(self.sv_model_path)
            
            # save new configration
            with open(self.config_file, 'w') as ymlfile:
                yaml.dump(self.cfg, ymlfile)
                
            print("Step 8: Check if the new model is better than last saved model --> Yes")
                
        else:
            print("Step 8: Check if the new model is better than last saved model --> No")
            
        #save the new score
        self.save_scores()
                
        
    # -----------------------------------------------------------------

    
    def save_scores(self):
        # get score for each class
        category_classes_scores = self.get_class_score(self.test_ul_ds)
        
        # save the new score
        new_score = {'id': 1 if self.scores.shape[0]==0 else self.scores['id'].max()+1,
                     'train_set':self.train_ul_ds.num_rows,
                     'test_set':self.test_ul_ds.num_rows,
                     'score':self.last_score, 
                     'creation_date':date.today()}
        
        for lab, score in zip(self.labels, category_classes_scores):
            new_score[lab] = score
            
        #append row to the dataframe
        self.scores = self.scores.append(new_score, ignore_index=True)
        file = open(self.score_path,'wb')
        pickle.dump(self.scores, file)
        
         
        
    # -----------------------------------------------------------------
    
      
    def predict_category(self, text_lst):
        pred_labels_id = self.model(text_lst)
        pred_labels = list(map(lambda x: self.labels[x], pred_labels_id))
        pred_labels_prob = [round(prob[l_id].item(), 2) for prob, l_id in zip(self.model.predict_proba(text_lst), pred_labels_id)]
        return pred_labels_id, pred_labels, pred_labels_prob
         
        
    # -----------------------------------------------------------------
    
      
    def get_class_score(self, test_ds):
        
        # get prediction for test data
        pred_labels_id, pred_labels, pred_labels_prob = self.predict_category(test_ds['text'])
        true_label = test_ds['label']
        
        # get scores
        matrix = confusion_matrix(true_label, pred_labels_id)
        return matrix.diagonal()/matrix.sum(axis=1)
    
    
         
        
    # -----------------------------------------------------------------
    
      
    def improve_model(self, condition):
        if self.first_label_flag:
            print("Start an iteration ------ First iteration")
            return True
        if self.ac == 'Overall accuracy':
            last_score = self.scores['score'].max()
            if condition >= last_score:
                print(f"Start an iteration - last score {last_score} is less than the target score {condition}")
                return True
            else:
                print(f"Don't Start an iteration - last score {last_score} is greater than or equal the target score {condition}")
                return False
        elif self.ac == 'iterations':
            last_itr = self.scores['id'].max()
            if condition > last_itr:
                print(f"Start an iteration - number of  iterations {last_itr} is less than the target iterations {condition}")
                return True
            else:
                print(f"Don't Start an iteration - number of  iterations {last_itr} is greater than or equal the target iterations {condition}")
                return False
        else:#'Per category accuracy'
            for label in self.labels:
                last_score = self.scores[label].max()
                if condition >= last_score:
                    print(f"Start an iteration - last score {last_score} for category {label} is less than the target score {condition}")
                    return True
            print(f"Don't Start an iteration - last scores for all categories are greater than or equal the target score {condition}")
            return False

    
         
        
    # -----------------------------------------------------------------        