from label_model import BuildFewShotModel
import time


improve_condition = 90
data_row_limit = 15
reuse_current_data = False

# create listner object and set some variables
model_obj = BuildFewShotModel(config_file="../config.yml",
                              data_file_name="Restaurant_Reviews",
                              active_learning_method=1,
                              business_kpi="argilla.apikey")

##-----------Active learning approch-------------------------------##
# try:
#     while(model_obj.improve_model(condition=improve_condition)):
#         if not reuse_current_data and not model_obj.has_labeling_running():
#             data = model_obj.load_data_pool()
#             model_obj.prepare_data(row_limit=data_row_limit) 
#             # model_obj.upload_data_to_UI()
#         while(not model_obj.is_finish_labeling(limit_row=data_row_limit)):
#             time.sleep(5)
#         model_obj.load_labeled_data()
#         print(model_obj.train_model())
# except KeyboardInterrupt:
#     pass

##-----------One Run approch-------------------------------##
# step 3: labeling
if not model_obj.has_labeling_running():
    data = model_obj.load_data_pool()
    model_obj.prepare_data(row_limit=data_row_limit) 
    model_obj.upload_data_to_UI()
# step 4: train model
if model_obj.is_finish_labeling(limit_row=data_row_limit):
    model_obj.load_labeled_data()   
    print(model_obj.train_model())

