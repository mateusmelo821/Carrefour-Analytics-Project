import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

df_p1 = pd.read_csv("https://raw.githubusercontent.com/mateusmelo821/Carrefour-Analytics-Project/main/data/data_p1.csv")
df_p2 = pd.read_csv("https://raw.githubusercontent.com/mateusmelo821/Carrefour-Analytics-Project/main/data/data_p2.csv")
df = pd.concat([df_p1, df_p2], ignore_index=True)

df_grouped = df.groupby('item_descricao')['ticket'].count()
df_grouped = df_grouped.sort_values('ticket', ascending=False, ignore_index=True).head(100)
df = df[df['item_descricao'].isin(df_grouped['item_descricao'])]
df = df[['ticket', 'item_descricao']]

df_format = pd.crosstab(df['ticket'], df['item_descricao'])
df_format = df_format>0

df_suporte = apriori(df_format, min_support=0.003, use_colnames=True)
df_combos = association_rules(df_suporte, metric='lift', min_threshold=1)[['antecedents', 'consequents', 'antecedent support',
                                                                           'support', 'confidence', 'lift']]

df_combos[df_combos['antecedents'].apply(lambda x: len(x)>1)].to_csv("resultado_combos_3_produtos.csv", index=False)

df_combos_2 = df_combos[(df_combos['antecedents'].apply(lambda x: len(x)==1))&(df_combos['consequents'].apply(lambda x: len(x)==1))]
df_combos_2['antecedents'] = df_combos_2['antecedents'].apply(lambda x: list(x)[0])
df_combos_2['consequents'] = df_combos_2['consequents'].apply(lambda x: list(x)[0])
df_combos_2.to_csv('produtos_relevantes.csv', index=False)