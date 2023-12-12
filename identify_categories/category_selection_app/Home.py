import streamlit as st
# import time
import pickle
import model

# general configuration for dataset 
file_path = '../../data/One45_comments.xlsx'
text_col = 'comment :'
folder_name = file_path.split('/')[-1].split('.')[0]
model_path = f"../../models/{folder_name}/Unsupervised_model"
config_path = f"../../config.yml"
dataset_name = "One45_comments"



# Load Model and set the required variabels
with st.spinner('Wait for loading model and data ...'):   
    if 'model' not in st.session_state:
        st.session_state.model = model.BertStreamlit(model_file=f'{model_path}/model_01',
                                                 parameter_file=f'{model_path}/model_parameters.pkl')
        
        # get the saved model separately for easy calling later 
        st.session_state.model_obj = getattr(st.session_state.model,'topic_model')
        # get the text
        st.session_state.text_lst = getattr(st.session_state.model,'data_lang')['en'][text_col].values
        # get extracted unique topics
        st.session_state.topics_df = st.session_state.model_obj.get_topic_info()
        st.session_state.topics_df = st.session_state.topics_df[st.session_state.topics_df['Topic']!=-1]
        # set limit for accepted topics
        st.session_state.text_count = 10
        # get hierarchical viz
#         st.session_state.h_fig = st.session_state.model.get_hierarchical_topics_chart(st.session_state.text_lst,
#                                                                                       fig_file=f'{model_path}/h_fig.pkl')
        # get documents viz
        st.session_state.d_fig = st.session_state.model.get_visualize_documents_chart(st.session_state.text_lst)

# main header
st.info('### What your data taked about')
st.markdown("""---""")

# side bar menu
with st.sidebar:
    step = st.radio(
        "Explore topics in your data:",
        ("1: Select a topic",
         "2: Display top words belong to the topic",
         "3: Display text belong to the topic",
         "4: show topic relationships",
         "5: Submit the topic")
    )
    # page logo
    st.markdown("""---""")
    st.image("images/scfhs-logo.png")


# content for each step
if step == "1: Select a topic":    
    # selector box for all extracted topics
    topics_lst = st.session_state.topics_df[st.session_state.topics_df['Count'] > st.session_state.text_count]['CustomName'].values
    st.session_state.topic = st.selectbox('Topics:',topics_lst)
    st.markdown("""---""")
    # controler bar to filter topics based on how many comments belong to it
    col1, col2, col3 = st.columns(3)
    st.session_state.text_count = col2.slider('The number of comments belong to each topic should be larger than:',
                                            int(st.session_state.topics_df['Count'].min()),
                                            int(st.session_state.topics_df['Count'].max()),
                                            st.session_state.text_count)
    # get id of selected topic
    st.session_state.topic_id = int(st.session_state.topics_df[st.session_state.topics_df['CustomName']==st.session_state.topic]['Topic'])
# -------------------------------------------------------------
elif step == "2: Display top words belong to the topic": 
    fig = st.session_state.model.get_top_words(st.session_state.topic_id)
    st.plotly_chart(fig, theme=None, use_container_width=True)
#-------------------------------------------------------------
elif step == "3: Display text belong to the topic":
    for trace in st.session_state.d_fig['data']:
        if trace['name'] != st.session_state.topic:
            trace['visible'] ='legendonly'
        else:
            trace['visible'] = None
    st.plotly_chart(st.session_state.d_fig, theme=None, use_container_width=True)
# #-------------------------------------------------------------
elif step == "4: show topic relationships":
#     st.plotly_chart(st.session_state.h_fig, theme=None, use_container_width=True)
    st.image(f'{model_path}/h_fig.png')
#-------------------------------------------------------------
else:
    # select how to save the selected topic
    submit = st.radio("",
        ("Add new Topic/Category", "Link with exist topic/category"))
    if submit == "Add new Topic/Category":
        topic_name = st.text_input('Topic/Category label', '')
    else:
        saved_topics = st.session_state.model.get_saved_topic(config_path,
                                                              dataset_name)
        if saved_topics == 0 :
            st.warning('There is no saved topics/categories before, please add one first!', icon="⚠️")
        else:
            topic_name = st.selectbox('',set(saved_topics.values()))

    st.markdown("""---""")
    if st.button('Save') and topic_name != '':
        st.success(st.session_state.model.save_topic(config_path,
                                                   st.session_state.topic_id,
                                                   topic_name,
                                                   dataset_name)
                   , icon="✅")
    

        
