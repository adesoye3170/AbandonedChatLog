import nltk
nltk.download('words')
nltk.download('punkt')
nltk.download('stopwords')
import conversation_analytics_toolkit
from conversation_analytics_toolkit import wa_assistant_skills
from conversation_analytics_toolkit import transformation
from conversation_analytics_toolkit import filtering2 as filtering
from conversation_analytics_toolkit import analysis 
from conversation_analytics_toolkit import visualization 
from conversation_analytics_toolkit import selection as vis_selection
from conversation_analytics_toolkit import wa_adaptor 
from conversation_analytics_toolkit import transcript 
from conversation_analytics_toolkit import flows 
from conversation_analytics_toolkit import keyword_analysis 
from conversation_analytics_toolkit import sentiment_analysis 

import json
import pandas as pd
from pandas.io.json import json_normalize
from IPython.core.display import display, HTML
import ibm_watson
from ibm_watson import AssistantV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from flask import Flask, render_template, request, url_for
import datetime
import pytz


app = Flask(__name__)


def spool_data():
    pd.set_option('display.max_colwidth', -1)

    authenticator = IAMAuthenticator("9nJ7-MvPt9AVtK72-YotVNLRLvGARrEc3lls1oTgO1MW")
    service = AssistantV1(version='2019-02-28',authenticator = authenticator)
    service.set_service_url("https://gateway.watsonplatform.net/assistant/api")

    #select a workspace by specific id
    workspace_id = '61d70d27-a82d-495a-b6dc-d7b129a651ce'
    # or fetch one via the APIs
    # workspaces=service.list_workspaces().get_result()
    # workspace_id = service['workspaces'][0]['workspace_id']

    #fetch the workspace
    workspace=service.get_workspace(
        workspace_id=workspace_id,
        export=True
    ).get_result()

    # set query parameters
    limit_number_of_records=40000
    # example of time range query
    query_filter = "response_timestamp>=2021-04-23,response_timestamp<2021-04-24"
    #query_filter = None
    # Fetch the logs for the workspace
    df_logs = wa_adaptor.read_logs(service, workspace_id, limit_number_of_records, query_filter)

    skill_id = workspace_id
    assistant_skills = wa_assistant_skills.WA_Assistant_Skills()
    assistant_skills.add_skill(skill_id, workspace)
    #validate the number of workspace_ids
    print("workspace_ids in skills: " + pd.DataFrame(assistant_skills.list_skills())["skill_id"].unique())
    print("workspace_ids in logs: "+ df_logs.workspace_id.unique())

    df_logs_canonical = transformation.to_canonical_WA_v2(df_logs, assistant_skills, skill_id_field=None, include_nodes_visited_str_types=True, include_context=False)
    #df_logs_canonical = transformation.to_canonical_WA_v2(df_logs, assistant_skills, skill_id_field="workspace_id", include_nodes_visited_str_types=True, include_context=False)

    # the rest of the notebook runs on the df_logs_to_analyze object.  
    df_logs_to_analyze = df_logs_canonical.copy(deep=False)
    

    '''title = "All Conversations"
    turn_based_path_flows = analysis.aggregate_flows(df_logs_to_analyze, mode="turn-based", on_column="turn_label", max_depth=400, trim_reroutes=False)
    # increase the width of the Jupyter output cell   
    display(HTML("<style>.container { width:95% !important; }</style>"))
    config = {
        'commonRootPathName': title, # label for the first root node 
        'height': 800, # control the visualization height.  Default 600
        'nodeWidth': 250, 
        'maxChildrenInNode':10, # control the number of immediate children to show (and collapse rest into *others* node).  Default 5
        'linkWidth' : 400,  # control the width between pathflow layers.  Default 360     'sortByAttribute': 'flowRatio'  # control the sorting of the chart. (Options: flowRatio, dropped_offRatio, flows, dropped_off, rerouted)
        'sortByAttribute': 'flowRatio',
        'title': title,
        'mode': "turn-based"
    }



    jsondata = json.loads(turn_based_path_flows.to_json(orient='records'))
    visualization.draw_flowchart(config, jsondata, python_selection_var="selection")

    # filter the conversations that include escalation
    title2="Banking Card Escalated"
    filters = filtering.ChainFilter(df_logs_to_analyze).setDescription(title2) 
    # node with condition on the #Banking-Card_Selection (node_1_1510880732839) and visit the node "Transfer To Live Agent" (node_25_1516679473977)
    filters.by_dialog_node_id('node_1_1537359291884')\
        .by_dialog_node_id('node_2_1557410657039')\
        .by_dialog_node_id('node_2_1557485570085')\
        .by_dialog_node_id('node_1_1539687257421')\
        #.by_dialog_node_id('node_4_1550388155701')\
        #.by_dialog_node_id('node_4_1550388155701')
        #.by_dialog_node_id('node_8_1591090496445')
        
        



    filters.printConversationFilters() 
    # get a reference to the dataframe.  Note: you can get access to intermediate dataframes by calling getDataFrame(index)

    ##define the milestones and corresponding node ids for the `Schedule Appointment` task
    milestone_analysis = analysis.MilestoneFlowGraph(assistant_skills.get_skill_by_id(skill_id))

    milestone_analysis.add_milestones(["Appointment scheduling start", "Enter purpose of appointment", "Scheduling completion", "Enter Zip Code", "Schedule time"])

    milestone_analysis.add_node_to_milestone("node_1_1537359291884", "Appointment scheduling start")   
    milestone_analysis.add_node_to_milestone("node_2_1557410657039", "Enter purpose of appointment")
    milestone_analysis.add_node_to_milestone("node_2_1557485570085", "Scheduling completion")
    milestone_analysis.add_node_to_milestone("node_4_1550388155701", "Enter Zip Code")
    milestone_analysis.add_node_to_milestone("node_1_1539687257421", "Schedule time")

    #enrich with milestone information - will add a column called 'milestone'
    milestone_analysis.enrich_milestones(df_logs_to_analyze)
    #remove all log records without a milestone
    df_milestones = df_logs_to_analyze[pd.isna(df_logs_to_analyze["milestone"]) == False]
    #optionally, remove consecutive milestones for a more simplified flow visualization representation
    df_milestones = analysis.simplify_flow_consecutive_milestones(df_milestones)

    # compute the aggregate flows of milestones 
    computed_flows= analysis.aggregate_flows(df_milestones, mode="milestone-based", on_column="milestone", max_depth=30, trim_reroutes=False)
    config = {
        'commonRootPathName': 'All Conversations', # label for the first root node 
        'height': 800, # control the visualization height.  Default 600
        'maxChildrenInNode': 6, # control the number of immediate children to show (and collapse the rest into *other* node).  Default 5
    #     'linkWidth' : 400,  # control the width between pathflow layers.  Default 360     '
        'sortByAttribute': 'flowRatio', # control the sorting of the chart. (Options: flowRatio, dropped_offRatio, flows, dropped_off, rerouted)
        'title': "Abandoned Conversations in Appointment Schedule Flow",
        'showVisitRatio' : 'fromTotal', # default: 'fromTotal'.  'fromPrevious' will compute percentages from previous step,
        'mode': 'milestone-based'
    }
    jsondata = json.loads(computed_flows.to_json(orient='records'))
    visualization.draw_flowchart(config, jsondata, python_selection_var="milestone_selection")
    #the selection variable contains details about the selected node, and conversations that were abandoned at that point
    print("Selected Path: ", jsondata)
    #fetch the dropped off conversations from the selection
    # 

 
    #dropped_off_conversations = vis_selection.to_dataframe(jsondata)["dropped_off"]
    realData = jsondata
    original = [data['dropped_off'] for data in realData]    

    dropped_off_conversations = json.loads(json.dumps(original))
    print("The selection contains {} records, with a reference back to the converstion logs".format(str(len(dropped_off_conversations))))


    

    print("start")
    print(df_logs_to_analyze["request_text"])
    #try:
    #    df_logs_to_analyze=TextBlob(df_logs_to_analyze).sentiment
    #    df_logs_to_analyze = sentiment_analysis.add_sentiment_columns(df_logs_to_analyze)
    #except:
    #    df_logs_to_analyze=sentiment_analysis.add_sentiment_columns(df_logs_to_analyze)
    print("end")
    df_logs_to_analyze["request_text"]= df_logs_to_analyze["request_text"].astype(str)
    df_logs_to_analyze = sentiment_analysis.add_sentiment_columns(df_logs_to_analyze) 
    #create insights, and highlights annotation for the transcript visualization
    NEGATIVE_SENTIMENT_THRESHOLD=-0.15 
    df_logs_to_analyze["insights_tags"] = df_logs_to_analyze.apply(lambda x: ["Negative Sentiment"] if x.sentiment < NEGATIVE_SENTIMENT_THRESHOLD else [], axis=1)
    df_logs_to_analyze["highlight"] = df_logs_to_analyze.apply(lambda x: True if x.sentiment < NEGATIVE_SENTIMENT_THRESHOLD else False, axis=1)'''

    newData = df_logs_to_analyze.loc[:, ["conversation_id", "log_id", "request_text", "response_text", "intent_1"]]
    result = newData.to_json(orient="table")

    data = json.loads(json.dumps(result, indent=4))

    
    print(df_logs_to_analyze)
    return data
    
@app.route('/data', methods=['GET'])
def data():
    resp = spool_data()
    return resp