import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from engine import StorageManager
import os

# Ensure you have OPENAI_API_KEY in your env or .env file
# If no key is present, we fallback to simple SQL translation logic for the demo

class TransformationAgent:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.storage = StorageManager()

    def apply_business_rule(self, df: pd.DataFrame, rule_description: str, rule_name: str) -> pd.DataFrame:
        """
        Applies a transformation based on a natural language rule.
        """
        print(f"Executing Agent Job: {rule_name}...")
        
        if self.api_key:
            # AI MODE: Use LLM to figure out the transformation
            try:
                llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", openai_api_key=self.api_key)
                agent = create_pandas_dataframe_agent(
                    llm, 
                    df, 
                    verbose=True,
                    agent_type=AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code=True # Enabled for demo purposes
                )
                # We ask the agent to return the modified dataframe code or description
                # For this demo wrapper, we will ask it to output the result directly
                # Note: Extracting the DF back from the agent is tricky, 
                # so we will use the agent to generate code and exec it, 
                # or simpler: use the agent to filter/clean and save to CSV then reload.
                
                # Alternate Strategy for Stability: Text-to-SQL via LLM
                prompt = f"""
                You are a data engineer. given the dataframe with columns {df.columns},
                write a DuckDB SQL query to transform the data according to this rule: "{rule_description}".
                The table name is 'CURRENT_TABLE'. Return ONLY the SQL string.
                """
                response = llm.predict(prompt)
                sql_query = response.strip().replace("```sql", "").replace("```", "")
                return self.storage.execute_sql(sql_query, "temp_staging") 
                
            except Exception as e:
                print(f"AI Agent failed, falling back to pass-through: {e}")
                return df
        else:
            # MANUAL MODE (Fallback if no API Key): specific keyword mapping
            # This makes the demo work even without internet/credits
            if "filter" in rule_description.lower():
                # specific hardcoded logic for demo safety
                return df.head(10) # Mock transformation
            return df

    def get_rule_dictionary(self):
        """Returns the DAG/Graph of available rules."""
        return {
            "clean_emails": "Remove rows with invalid email formats",
            "standardize_currency": "Convert all revenue columns to USD",
            "remove_outliers": "Remove z-score > 3 in numeric columns",
            "top_performers": "Filter top 10% by sales"
        }
