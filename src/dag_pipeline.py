from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
from datetime import datetime
from mlxtend.frequent_patterns import apriori, association_rules

def pipeline():
  credentials = service_account.Credentials.from_service_account_file('credentials.json')
  project_id = 'project-id-123'
  client = bigquery.Client(credentials=credentials,project=project_id)
  Q1_query = ''
  date_max = str(datetime.today())[:10]
  date_min = str(datetime.today()-pd.Timedelta(days=3))[:10]
  with open("Q1.sql", 'r') as arquivo:
    Q1_query = arquivo.read()
  df = pd.read_gbq(Q1_query.format(date_max=date_max, date_min=date_min), project_id=project_id, credentials=credentials)

  df_grouped = df.groupby('item_descricao')['ticket'].count()
  df_grouped = df_grouped.sort_values('ticket', ascending=False, ignore_index=True).head(100)
  df = df[df['item_descricao'].isin(df_grouped['item_descricao'])]
  df = df[['ticket', 'item_descricao']]
  df_format = pd.crosstab(df['ticket'], df['item_descricao'])
  df_format = df_format>0
  df_suporte = apriori(df_format, min_support=0.003, use_colnames=True)
  df_combos = association_rules(df_suporte, metric='lift', min_threshold=1)[['antecedents', 'consequents', 'antecedent support',
                                                                           'support', 'confidence', 'lift']]

  df_combos[df_combos['antecedents'].apply(lambda x: len(x)>1)].to_gbq('combos_multiplos', project_id=project_id,
                                                                       credentials=credentials, if_exists='replace')
  df_combos_2 = df_combos[(df_combos['antecedents'].apply(lambda x: len(x)==1))&(df_combos['consequents'].apply(lambda x: len(x)==1))]
  df_combos_2['antecedents'] = df_combos_2['antecedents'].apply(lambda x: list(x)[0])
  df_combos_2['consequents'] = df_combos_2['consequents'].apply(lambda x: list(x)[0])
  df_combos_2.to_gbq('combos', project_id=project_id, credentials=credentials, if_exists='replace')

with DAG(
    dag_id='pipeline_dag',
    start_date=datetime(2024, 2, 15),
    retries=2,
    schedule_interval='0 7 * * 4'
) as dag:

  task_start = EmptyOperator(task_id='start')

  task_pipeline = PythonOperator(
      task_id='task_pipeline',
      dag=dag,
      python_callablle=pipeline
  )

  task_end = EmptyOperator(task_id='end')

  task_start >> task_pipeline >> task_end