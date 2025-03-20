#!/usr/bin/env python

# Copyright 2024 Amazon.com and its affiliates; all rights reserved.
# This file is AWS Content and may not be duplicated or distributed without permission
import sys
from pathlib import Path
import datetime
import traceback
import yaml
import uuid
from textwrap import dedent
import os
import argparse
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.utils.bedrock_agent import Agent, SupervisorAgent, Task, region, account_id
from src.utils.knowledge_base_helper import KnowledgeBasesForAmazonBedrock

kb_helper = KnowledgeBasesForAmazonBedrock()

current_dir = os.path.dirname(os.path.abspath(__file__))
task_yaml_path = os.path.join(current_dir, "tasks.yaml")
agent_yaml_path = os.path.join(current_dir, "agents.yaml")

def main(args):
    kb_name = "financial-advisor-kb"
    print(f"Acquiring  {kb_name} knowledge base")
    bucket_name = 'lhbank-ts-dev-curated-2'
    bucket_object_prefixs = ['kb-sources/Company Data/*.*']
    kb_id, ds_id = kb_helper.create_or_retrieve_knowledge_base(
        kb_name,
        kb_description="Useful for answering questions about customer loaning and for questions about the company annual financial reports",
        data_bucket_name=bucket_name,
        bucket_object_prefixs=bucket_object_prefixs 
    )
    bucket_name = kb_helper.data_bucket_name
    print(f"KB name: {kb_name}, kb_id: {kb_id}, ds_id: {ds_id}, s3 bucket name: {bucket_name}\n")

    if args.recreate_agents == "true":
        # sync knowledge base
        kb_helper.synchronize_data(kb_id, ds_id)
        print('KB sync completed\n')


    if args.recreate_agents == "false":
        Agent.set_force_recreate_default(False)
    else:
        Agent.set_force_recreate_default(True)
        Agent.delete_by_name("financial_advisor", verbose=True)
    if args.clean_up == "true":
        Agent.delete_by_name("financial_advisor", verbose=True)
        Agent.delete_by_name("financial_internal_analyst", verbose=True)       
        Agent.delete_by_name("financial_external_analyst", verbose=True)
        Agent.delete_by_name("formatted_report_writer", verbose=True)
        kb_helper.delete_kb("financial-advisor-kb", delete_s3_bucket=False)
        
    else:
        inputs = {
            'company_name': args.company_name,
            'feedback_iteration_count': args.iterations,
        }    

        with open(task_yaml_path, 'r') as file:
            task_yaml_content = yaml.safe_load(file)

        finacial_extract_all_task = Task('finacial_extract_all_task', task_yaml_content, inputs)
        finacial_get_external_data_task = Task('finacial_get_external_data_task', task_yaml_content, inputs)
        final_report_output_task = Task('final_report_output_task', task_yaml_content, inputs)
            
        web_search_tool = {
            "code":f"arn:aws:lambda:{region}:{account_id}:function:web_search",
            "definition":{
                "name": "web_search",
                "description": "Searches the web for information",
                "parameters": {
                    "search_query": {
                        "description": "The query to search the web with",
                        "type": "string",
                        "required": True,
                    },
                    "target_website": {
                        "description": "The specific website to search including its domain name. If not provided, the most relevant website will be used",
                        "type": "string",
                        "required": False,
                    },
                    "topic": {
                        "description": "The topic being searched. 'news' or 'general'. Helps narrow the search when news is the focus.",
                        "type": "string",
                        "required": False,
                    },
                    "days": {
                        "description": "The number of days of history to search. Helps when looking for recent events or news.",
                        "type": "string",
                        "required": False,
                    },
                },
            },
        }

        set_value_for_key = {
            "code":f"arn:aws:lambda:{region}:{account_id}:function:working_memory",
            "definition":{
                "name": "set_value_for_key",
                "description": " Stores a key-value pair in a DynamoDB table. Creates the table if it doesn't exist.",
                "parameters": {
                    "key": {
                        "description": "The name of the key to store the value under.",
                        "type": "string",
                        "required": True,
                    },
                    "value": {
                        "description": "The value to store for that key name.",
                        "type": "string",
                        "required": True,
                    },
                    "table_name": {
                        "description": "The name of the DynamoDB table to use for storage.",
                        "type": "string",
                        "required": True,
                    }
                },
            },
        }

        get_key_value = {
            "code":f"arn:aws:lambda:{region}:{account_id}:function:working_memory",
            "definition":{
                "name": "get_key_value",
                "description": "Retrieves a value for a given key name from a DynamoDB table.",
                "parameters": {
                    "key": {
                        "description": "The name of the key to store the value under.",
                        "type": "string",
                        "required": True,
                    },
                    "table_name": {
                        "description": "The name of the DynamoDB table to use for storage.",
                        "type": "string",
                        "required": True,
                    }
                },
            },
        }

        with open(agent_yaml_path, 'r') as file:
            agent_yaml_content = yaml.safe_load(file)

        financial_internal_analyst = Agent('financial_internal_analyst', agent_yaml_content,
                                    tools=[web_search_tool, set_value_for_key, get_key_value])
        financial_external_analyst = Agent('financial_external_analyst', agent_yaml_content,
                                    tools=[web_search_tool, set_value_for_key, get_key_value])
        formatted_report_writer = Agent('formatted_report_writer', agent_yaml_content,
                                  tools=[web_search_tool, set_value_for_key, get_key_value])
        
        print("\n\nCreating financial_strategy_agent as a supervisor agent...\n\n")
        financial_advisor = SupervisorAgent("financial_advisor", agent_yaml_content,
                                    [financial_internal_analyst, financial_external_analyst, 
                                    formatted_report_writer], 
                                    verbose=False)
        
        if args.recreate_agents == "false":
            print("\n\nInvoking supervisor agent...\n\n")

            time_before_call = datetime.datetime.now()
            print(f"time before call: {time_before_call}\n")
            try:
                folder_name = "financial-advisor-" + str(uuid.uuid4())
                result = financial_advisor.invoke_with_tasks([
                                finacial_extract_all_task, finacial_get_external_data_task,
                                final_report_output_task
                            ],
                            additional_instructions=dedent(f"""
                                Use a single Working Memory table for this entire set of tasks, with 
                                table name: {folder_name}. Tell your collaborators this table name as part of 
                                every request, so that they are not confused and they share state effectively.
                                The keys they use in that table will allow them to keep track of any number 
                                of state items they require. When you have completed all tasks, summarize 
                                your work, and share the table name so that all the results can be used and 
                                analyzed."""),
                            processing_type="sequential", 
                            enable_trace=True, trace_level=args.trace_level,
                            verbose=True)
                print(result)
            except Exception as e:
                print(e)
                traceback.print_exc()
                pass

            duration = datetime.datetime.now() - time_before_call
            print(f"\nTime taken: {duration.total_seconds():,.1f} seconds")
        else:
            print("Recreated agents.")
        


if __name__ == '__main__':

    default_inputs = {
        'company_name': 'บริษัท กมลโลหะกิจ จำกัด',
        'iterations': "1"
    }    

    parser = argparse.ArgumentParser()

    parser.add_argument("--recreate_agents", required=False, default='true', help="False if reusing existing agents.")
    parser.add_argument("--company_name", required=False, 
                        default=default_inputs['company_name'],
                        help="The company name for analysis")
    parser.add_argument("--iterations", required=False, 
                        default=default_inputs['iterations'],
                        help="The number of rounds of feedback to use when producing the analysis report")
    parser.add_argument("--trace_level", required=False, default="core", help="The level of trace, 'core', 'outline', 'all'.")
    parser.add_argument(
        "--clean_up",
        required=False,
        default="false",
        help="Cleanup all infrastructure.",
    )
    args = parser.parse_args()
    main(args)
