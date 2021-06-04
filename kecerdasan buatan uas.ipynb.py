# Load in our libraries
import pandas as pd
import numpy as np
import re
import sklearn
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings
warnings.filterwarnings('ignore')

# Going to use these 5 base models for the stacking
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC
from sklearn.model_selection import KFold
Em progesso..
# settings
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
df1 = pd.read_csv('../input/credit-card-approval-prediction/application_record.csv')
df2 = pd.read_csv('../input/credit-card-approval-prediction/credit_record.csv')
df1.head(5)
ID	CODE_GENDER	FLAG_OWN_CAR	FLAG_OWN_REALTY	CNT_CHILDREN	AMT_INCOME_TOTAL	NAME_INCOME_TYPE	NAME_EDUCATION_TYPE	NAME_FAMILY_STATUS	NAME_HOUSING_TYPE	DAYS_BIRTH	DAYS_EMPLOYED	FLAG_MOBIL	FLAG_WORK_PHONE	FLAG_PHONE	FLAG_EMAIL	OCCUPATION_TYPE	CNT_FAM_MEMBERS
0	5008804	M	Y	Y	0	427500.0	Working	Higher education	Civil marriage	Rented apartment	-12005	-4542	1	1	0	0	NaN	2.0
1	5008805	M	Y	Y	0	427500.0	Working	Higher education	Civil marriage	Rented apartment	-12005	-4542	1	1	0	0	NaN	2.0
2	5008806	M	Y	Y	0	112500.0	Working	Secondary / secondary special	Married	House / apartment	-21474	-1134	1	0	0	0	Security staff	2.0
3	5008808	F	N	Y	0	270000.0	Commercial associate	Secondary / secondary special	Single / not married	House / apartment	-19110	-3051	1	0	1	1	Sales staff	1.0
4	5008809	F	N	Y	0	270000.0	Commercial associate	Secondary / secondary special	Single / not married	House / apartment	-19110	-3051	1	0	1	1	Sales staff	1.0
df2.head(2)
ID	MONTHS_BALANCE	STATUS
0	5001711	0	X
1	5001711	-1	0
limpando os dados
# Verificando a presença de valores nulos no data frame
df1.isnull().sum().sum()
134203
# Verificando a presença de valores nulos no data frame
df2.isnull().sum().sum()
0
# Verificando a presença de NA's no data frame
df1.isnull().values.any()
True
#REmovendo os Null/NA
df1 = df1.dropna()
#verificando se sobrou algum Null/NA
df1.isnull().values.any()
False
Verificando o numero de linhas restantes
index = df1.index
number_of_rows = len(index)
print(number_of_rows)
304354
index = df2.index
number_of_rows = len(index)
print(number_of_rows)
1048575
Juntando as duas bases de dados
df3 = pd.merge(df1, df2, on='ID')
df3.head(3)
ID	CODE_GENDER	FLAG_OWN_CAR	FLAG_OWN_REALTY	CNT_CHILDREN	AMT_INCOME_TOTAL	NAME_INCOME_TYPE	NAME_EDUCATION_TYPE	NAME_FAMILY_STATUS	NAME_HOUSING_TYPE	DAYS_BIRTH	DAYS_EMPLOYED	FLAG_MOBIL	FLAG_WORK_PHONE	FLAG_PHONE	FLAG_EMAIL	OCCUPATION_TYPE	CNT_FAM_MEMBERS	MONTHS_BALANCE	STATUS
0	5008806	M	Y	Y	0	112500.0	Working	Secondary / secondary special	Married	House / apartment	-21474	-1134	1	0	0	0	Security staff	2.0	0	C
1	5008806	M	Y	Y	0	112500.0	Working	Secondary / secondary special	Married	House / apartment	-21474	-1134	1	0	0	0	Security staff	2.0	-1	C
2	5008806	M	Y	Y	0	112500.0	Working	Secondary / secondary special	Married	House / apartment	-21474	-1134	1	0	0	0	Security staff	2.0	-2	C
#Verificando a quantidade de dados depois da junção dos conjuntos
index = df3.index
number_of_rows = len(index)
print(number_of_rows)
537667
É possível notar que a base de dados aumentou, portanto houve duplicatas de valores. Precisamos remove-lás.
#verificando se sobrou algum Null/NA
df3.isnull().values.any()
False
#Tirando os ID duplicados
df3 = df3.drop_duplicates('ID',keep='first')
#Verificando a quantidade de dados depois da junção dos conjuntos
df3
ID	CODE_GENDER	FLAG_OWN_CAR	FLAG_OWN_REALTY	CNT_CHILDREN	AMT_INCOME_TOTAL	NAME_INCOME_TYPE	NAME_EDUCATION_TYPE	NAME_FAMILY_STATUS	NAME_HOUSING_TYPE	DAYS_BIRTH	DAYS_EMPLOYED	FLAG_MOBIL	FLAG_WORK_PHONE	FLAG_PHONE	FLAG_EMAIL	OCCUPATION_TYPE	CNT_FAM_MEMBERS	MONTHS_BALANCE	STATUS
0	5008806	M	Y	Y	0	112500.0	Working	Secondary / secondary special	Married	House / apartment	-21474	-1134	1	0	0	0	Security staff	2.0	0	C
30	5008808	F	N	Y	0	270000.0	Commercial associate	Secondary / secondary special	Single / not married	House / apartment	-19110	-3051	1	0	1	1	Sales staff	1.0	0	0
35	5008809	F	N	Y	0	270000.0	Commercial associate	Secondary / secondary special	Single / not married	House / apartment	-19110	-3051	1	0	1	1	Sales staff	1.0	-22	X
40	5008810	F	N	Y	0	270000.0	Commercial associate	Secondary / secondary special	Single / not married	House / apartment	-19110	-3051	1	0	1	1	Sales staff	1.0	0	C
67	5008811	F	N	Y	0	270000.0	Commercial associate	Secondary / secondary special	Single / not married	House / apartment	-19110	-3051	1	0	1	1	Sales staff	1.0	0	C
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
537574	5149828	M	Y	Y	0	315000.0	Working	Secondary / secondary special	Married	House / apartment	-17348	-2420	1	0	0	0	Managers	2.0	0	5
537586	5149834	F	N	Y	0	157500.0	Commercial associate	Higher education	Married	House / apartment	-12387	-1325	1	0	1	1	Medicine staff	2.0	0	C
537610	5149838	F	N	Y	0	157500.0	Pensioner	Higher education	Married	House / apartment	-12387	-1325	1	0	1	1	Medicine staff	2.0	0	C
537643	5150049	F	N	Y	0	283500.0	Working	Secondary / secondary special	Married	House / apartment	-17958	-655	1	0	0	0	Sales staff	2.0	0	2
537653	5150337	M	N	Y	0	112500.0	Working	Secondary / secondary special	Single / not married	Rented apartment	-9188	-1193	1	0	0	0	Laborers	1.0	0	0
25134 rows × 20 columns

Transformando em dummies
#Removendo a variável CODE_GENDER para não ter viés sexista na base de dados
df3 = df3.drop(columns=['CODE_GENDER'])
df3.head(3)
ID	FLAG_OWN_CAR	FLAG_OWN_REALTY	CNT_CHILDREN	AMT_INCOME_TOTAL	NAME_INCOME_TYPE	NAME_EDUCATION_TYPE	NAME_FAMILY_STATUS	NAME_HOUSING_TYPE	DAYS_BIRTH	DAYS_EMPLOYED	FLAG_MOBIL	FLAG_WORK_PHONE	FLAG_PHONE	FLAG_EMAIL	OCCUPATION_TYPE	CNT_FAM_MEMBERS	MONTHS_BALANCE	STATUS
0	5008806	Y	Y	0	112500.0	Working	Secondary / secondary special	Married	House / apartment	-21474	-1134	1	0	0	0	Security staff	2.0	0	C
30	5008808	N	Y	0	270000.0	Commercial associate	Secondary / secondary special	Single / not married	House / apartment	-19110	-3051	1	0	1	1	Sales staff	1.0	0	0
35	5008809	N	Y	0	270000.0	Commercial associate	Secondary / secondary special	Single / not married	House / apartment	-19110	-3051	1	0	1	1	Sales staff	1.0	-22	X
# Trasnformando todos de valores Y ou N em dummies, sendo 1 para Y
dummy1 = pd.get_dummies(df3.FLAG_OWN_CAR)
df3['FLAG_OWN_CAR'] = dummy1['Y']

dummy2 = pd.get_dummies(df3.FLAG_OWN_REALTY)
df3['FLAG_OWN_REALTY'] = dummy2['Y']


#Vendo as classes das variáveis categóricas
#print(df3['NAME_INCOME_TYPE'].unique())

#Vendo as classes das variáveis categóricas
#print(df3['NAME_EDUCATION_TYPE'].unique())

#Vendo as classes das variáveis categóricas
#print(df3['OCCUPATION_TYPE'].unique())

#Vendo as classes das variáveis categóricas
#print(df3['STATUS'].unique())
###criando uma variável ordinal para o nível de escolaridade
#df3['NAME_EDUCATION_TYPE'] =
 
Vamos tentar entender a capacidade de pagamento dos individuos e enquadrá-lo em categorias
#### Vamos ver como são os níveis de consumo por categorias sociais


fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
fig.suptitle('Consumo por característica')

# Bulbasaur
sns.barplot(ax=axes[0], x=df3.NAME_INCOME_TYPE, y=df3.AMT_INCOME_TOTAL).tick_params(labelrotation=45)
axes[0].set_title("Ocupação")


# Charmander
sns.barplot(ax=axes[1], x=df3.NAME_EDUCATION_TYPE, y=df3.AMT_INCOME_TOTAL).tick_params(labelrotation=45)
axes[1].set_title("Escolaridade")

# Squirtle
sns.barplot(ax=axes[2], x=df3.NAME_FAMILY_STATUS, y=df3.AMT_INCOME_TOTAL).tick_params(labelrotation=45)
axes[2].set_title("Estatus Civil")

#
sns.barplot(ax=axes[3], x=df3.NAME_HOUSING_TYPE, y=df3.AMT_INCOME_TOTAL).tick_params(labelrotation=45)
axes[3].set_title("Moradia")
Text(0.5, 1.0, 'Moradia')

###criando UM catplot individual para a variavel OCCUPATION_TYPE em relação ao poder de compra AMT_INCOME_TOTAL
plt.figure(figsize =(10,5))
ax = sns.barplot(x="OCCUPATION_TYPE", y="AMT_INCOME_TOTAL",data=df3).set_title('Consumo por profissão')
plt.xticks(rotation=60)
(array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17]),
 [Text(0, 0, 'Security staff'),
  Text(1, 0, 'Sales staff'),
  Text(2, 0, 'Accountants'),
  Text(3, 0, 'Laborers'),
  Text(4, 0, 'Managers'),
  Text(5, 0, 'Drivers'),
  Text(6, 0, 'Core staff'),
  Text(7, 0, 'High skill tech staff'),
  Text(8, 0, 'Cleaning staff'),
  Text(9, 0, 'Private service staff'),
  Text(10, 0, 'Cooking staff'),
  Text(11, 0, 'Low-skill Laborers'),
  Text(12, 0, 'Medicine staff'),
  Text(13, 0, 'Secretaries'),
  Text(14, 0, 'Waiters/barmen staff'),
  Text(15, 0, 'HR staff'),
  Text(16, 0, 'Realty agents'),
  Text(17, 0, 'IT staff')])

Como há várias categorias, dividiremos todas elas pelo poder de consumo
#média de consumo por profissão
df4 = df3.groupby(['OCCUPATION_TYPE']).mean().sort_values(['AMT_INCOME_TOTAL'], ascending=False)
df4['AMT_INCOME_TOTAL']
OCCUPATION_TYPE
Managers                 279117.292829
Realty agents            247500.000000
Drivers                  209797.240412
Accountants              202463.865834
IT staff                 199860.000000
Private service staff    198863.372093
High skill tech staff    196053.579176
HR staff                 193764.705882
Core staff               190172.786967
Laborers                 179794.282402
Security staff           177037.753378
Sales staff              174984.897848
Secretaries              168079.470199
Medicine staff           166114.618061
Waiters/barmen staff     156206.896552
Cleaning staff           149141.107078
Cooking staff            146517.251908
Low-skill Laborers       133920.000000
Name: AMT_INCOME_TOTAL, dtype: float64
# Como temos 18 profissões, vamos criar um indice de impacto de 6 níveis, de acordo com o poder de consumo
df3['OCCUPATION_TYPE'] = df3['OCCUPATION_TYPE'].replace(['Managers','Realty agents'],6)
df3['OCCUPATION_TYPE'] = df3['OCCUPATION_TYPE'].replace(['Drivers','Accountants','IT staff','Private service staff'],5)
df3['OCCUPATION_TYPE'] = df3['OCCUPATION_TYPE'].replace(['High skill tech staff','HR staff','Core staff','Laborers'],4)
df3['OCCUPATION_TYPE'] = df3['OCCUPATION_TYPE'].replace(['Security staff','Sales staff','Secretaries','Medicine staff'],3)
df3['OCCUPATION_TYPE'] = df3['OCCUPATION_TYPE'].replace(['Drivers','Accountants','IT staff','Private service staff'],2)
df3['OCCUPATION_TYPE'] = df3['OCCUPATION_TYPE'].replace(['Waiters/barmen staff','Cleaning staff','Cooking staff','Low-skill Laborers'],1)
df3['OCCUPATION_TYPE'] = df3['OCCUPATION_TYPE'].apply(pd.to_numeric)
# Fazendo o mesmo para educação
df5 = df3.groupby(['NAME_EDUCATION_TYPE']).mean().sort_values(['AMT_INCOME_TOTAL'], ascending=False)
df5['AMT_INCOME_TOTAL']
NAME_EDUCATION_TYPE
Academic degree                  253928.571429
Higher education                 229514.648345
Incomplete higher                202280.664653
Secondary / secondary special    179955.714570
Lower secondary                  165455.614973
Name: AMT_INCOME_TOTAL, dtype: float64
df3['NAME_EDUCATION_TYPE'] = df3['NAME_EDUCATION_TYPE'].replace(['Academic degree'],5)
df3['NAME_EDUCATION_TYPE'] = df3['NAME_EDUCATION_TYPE'].replace(['Higher education'],4)
df3['NAME_EDUCATION_TYPE'] = df3['NAME_EDUCATION_TYPE'].replace(['Incomplete higher'],3)
df3['NAME_EDUCATION_TYPE'] = df3['NAME_EDUCATION_TYPE'].replace(['Secondary / secondary special'],2)
df3['NAME_EDUCATION_TYPE'] = df3['NAME_EDUCATION_TYPE'].replace(['Lower secondary'],1)
df3['NAME_EDUCATION_TYPE'] = df3['NAME_EDUCATION_TYPE'].apply(pd.to_numeric)
print(df3['NAME_EDUCATION_TYPE'].unique())
[2 4 3 1 5]
# Fazendo o mesmo para finalidade de uso do crédito
df6 = df3.groupby(['NAME_INCOME_TYPE']).mean().sort_values(['AMT_INCOME_TOTAL'], ascending=False)
df6['AMT_INCOME_TOTAL']
NAME_INCOME_TYPE
Pensioner               257538.461538
Commercial associate    218450.592669
State servant           205066.709889
Working                 182547.168800
Student                 159300.000000
Name: AMT_INCOME_TOTAL, dtype: float64
df3['NAME_INCOME_TYPE'] = df3['NAME_INCOME_TYPE'].replace(['Pensioner'],5)
df3['NAME_INCOME_TYPE'] = df3['NAME_INCOME_TYPE'].replace(['Commercial associate'],4)
df3['NAME_INCOME_TYPE'] = df3['NAME_INCOME_TYPE'].replace(['State servant'],3)
df3['NAME_INCOME_TYPE'] = df3['NAME_INCOME_TYPE'].replace(['Working'],2)
df3['NAME_INCOME_TYPE'] = df3['NAME_INCOME_TYPE'].replace(['Student'],1)
df3['NAME_INCOME_TYPE'] = df3['NAME_INCOME_TYPE'].apply(pd.to_numeric)
print(df3['NAME_INCOME_TYPE'].unique())
[2 4 3 1 5]
 
# Fazendo o mesmo para finalidade de uso do crédito
df7 = df3.groupby(['NAME_HOUSING_TYPE']).mean().sort_values(['AMT_INCOME_TOTAL'], ascending=False)
df7['AMT_INCOME_TOTAL']
NAME_HOUSING_TYPE
Office apartment       237812.562814
Co-op apartment        222868.421053
Rented apartment       216431.825740
House / apartment      195017.475251
Municipal apartment    188764.470443
With parents           179850.883217
Name: AMT_INCOME_TOTAL, dtype: float64
df3['NAME_HOUSING_TYPE'] = df3['NAME_HOUSING_TYPE'].replace(['Office apartment'],6)
df3['NAME_HOUSING_TYPE'] = df3['NAME_HOUSING_TYPE'].replace(['Co-op apartment'],5)
df3['NAME_HOUSING_TYPE'] = df3['NAME_HOUSING_TYPE'].replace(['Rented apartment'],4)
df3['NAME_HOUSING_TYPE'] = df3['NAME_HOUSING_TYPE'].replace(['House / apartment'],3)
df3['NAME_HOUSING_TYPE'] = df3['NAME_HOUSING_TYPE'].replace(['Municipal apartment'],2)
df3['NAME_HOUSING_TYPE'] = df3['NAME_HOUSING_TYPE'].replace(['With parents'],1)
df3['NAME_HOUSING_TYPE'] = df3['NAME_HOUSING_TYPE'].apply(pd.to_numeric)
print(df3['NAME_HOUSING_TYPE'].unique())
[3 4 2 1 5 6]
 
#Como foi visto no gráfico inicial, a variável estado civil não varia muito de consumo de acordo como status,
# portanto vamos remove-lá junto as demais desnecessárias

df3 = df3.drop(columns=['NAME_FAMILY_STATUS'])
df3 = df3.drop(columns=['ID'])
#Vamos tira a variável FLAG_MOBIL , CNT_CHILDREN e FLAG_WORK_PHONE pois também não traz informação relevante
df3 = df3.drop(columns=['FLAG_MOBIL'])
df3 = df3.drop(columns=['FLAG_WORK_PHONE'])
df3 = df3.drop(columns=['CNT_CHILDREN'])
df3 = df3.drop(columns=['FLAG_PHONE'])
df3 = df3.drop(columns=['FLAG_EMAIL'])
df3.head(2)
FLAG_OWN_CAR	FLAG_OWN_REALTY	AMT_INCOME_TOTAL	NAME_INCOME_TYPE	NAME_EDUCATION_TYPE	NAME_HOUSING_TYPE	DAYS_BIRTH	DAYS_EMPLOYED	OCCUPATION_TYPE	CNT_FAM_MEMBERS	MONTHS_BALANCE	STATUS
0	1	1	112500.0	2	2	3	-21474	-1134	3	2.0	0	C
30	0	1	270000.0	4	2	3	-19110	-3051	3	1.0	0	0
Agora vamos analisar a variavel target
Temos duas categorias de indivíduos: com atrasos de pagamentos, e sem atrasos. Portanto vamos categorizá-los como inadimplentes e adimplentes. A decisão de quem é adimplente ou inadimplente é relativo e depende dos interesses internos das instituições, mas para simplificação do modelo fazeremos dessa forma.
Você deve estar se perguntando se não seria interessante estimar uma regressão linear antes de transformar a variável target em dummy. A questão é que as variáveis explicativas precisam ter distribuição normal para obter os melhores estimadores de MQO, o que não acontece no nosso conjunto de dados. Portando levaremos a nossa análise a modelos não linear.

#letras são adimplentes e números inadimplentes
df3['STATUS'] = df3['STATUS'].replace(['C'],0)
df3['STATUS'] = df3['STATUS'].replace(['X'],0)
df3['STATUS'] = df3['STATUS'].apply(pd.to_numeric) 
df3['STATUS'] = np.where(df3['STATUS']<1, 0, 1)
print(df3['STATUS'].unique())
[0 1]
Inadimplente = df3.loc[df3['STATUS'] == 1].count()[0]
Adimplente = df3.loc[df3['STATUS'] == 0].count()[0]

labels = ['days past due', 'paid off/No loan']
colors = ['#d10000', '#6297e3']
explode = (.1,.1)


plt.pie([Inadimplente, Adimplente], labels = labels, colors = colors, 
        autopct = '%.2f %%', pctdistance= 0.2, startangle=170, explode = explode)
plt.show()

Podemos notar que nosso conjunto de dados está muito desbalanceado e a proporção de classes é de 24853 Adimplentes para 281 Inadimplentes. E para isso vamos usar o método de resampling para balancear a base de dados.

import imblearn
from imblearn.under_sampling import RandomUnderSampler
x,y = df3.loc[:,df3.columns != 'STATUS'], df3.loc[:,'STATUS']

# Definindo a proporção de dados da classe onde há menos observações
sampling_strategy= 0.34
rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
X_res, y_res = rus.fit_resample(x, y)
autopct = "%.2f"
ax = y_res.value_counts().plot.pie(autopct=autopct)
_ = ax.set_title("Under-sampling")

 
# Chamar a variavel STATUS de risco ajuda a entener melhor
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(df3.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
<AxesSubplot:title={'center':'Correlation of Features'}>

As variáveis tem poucas correlação entre si, o que pode ser um sinal bom, diminuindo as chances de inflar o modelo.

Aplicando a técnica de ensemble stacking
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
# splitting the data

x_train,x_test,y_train,y_test = train_test_split(X_res,y_res,test_size = 0.20,random_state = 1)

x_train_0,x_train_1,y_train_0,y_train_1 = train_test_split(X_res,y_res,test_size = 0.60, random_state = 1)
#X_res, y_res
#criando uma lista com os modelos
# Vendo qual tem a melhor acurácia para usa-lo no stacking

models = {}
models['knn'] = KNeighborsClassifier()
models['cart'] = DecisionTreeClassifier()
models['svm'] = SVC()
models['bayes'] = GaussianNB()
models['rdm'] = RandomForestClassifier()
models['lgc'] = LogisticRegression(max_iter=1000)
models['ada'] = AdaBoostClassifier()
models['gda'] = GradientBoostingClassifier()
models['bca'] = BaggingClassifier()
# Voting method
# Método de votação
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(name, accuracy)
    
    #Suspeita de overfitting analisar pela curva roc
knn 0.7387387387387387
cart 0.6756756756756757
svm 0.7522522522522522
bayes 0.7477477477477478
rdm 0.8063063063063063
lgc 0.7522522522522522
ada 0.7387387387387387
gda 0.7792792792792793
bca 0.7882882882882883
Podemos ver que três modelos estão gerando overfitting, portanto iremos revome-los

# Rodando novamente
models = {}
models['knn'] = KNeighborsClassifier()
models['svm'] = SVC()
models['bayes'] = GaussianNB()
models['lgc'] = LogisticRegression(max_iter=1000)
models['ada'] = AdaBoostClassifier()
models['gda'] = GradientBoostingClassifier()

# Acurácias

# Voting method
# Método de votação
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(name, accuracy)

    
    
# Verificando o intervalo de confiança da acurácia
from sklearn.model_selection import cross_val_score    

knn_IC = cross_val_score(models['knn'], X_res,y_res, cv=5)
svm_IC = cross_val_score(models['svm'], X_res,y_res, cv=5)
bayes_IC = cross_val_score(models['bayes'], X_res,y_res, cv=5)
lgc_IC = cross_val_score(models['lgc'], X_res,y_res, cv=5)
gda_IC = cross_val_score(models['gda'], X_res,y_res, cv=5)
ada_IC = cross_val_score(models['ada'], X_res,y_res, cv=5)

scores = {}

scores['knn'] =  knn_IC.mean() + knn_IC.std() * 2, knn_IC.mean() - knn_IC.std() * 2
scores['svm'] =  svm_IC.mean() + svm_IC.std() * 2, svm_IC.mean() - svm_IC.std() * 2
scores['bayes'] =   bayes_IC.mean() + bayes_IC.std() * 2,bayes_IC.mean() - bayes_IC.std() * 2
scores['lgc'] =  lgc_IC.mean() + lgc_IC.std() * 2, lgc_IC.mean() - lgc_IC.std() * 2
scores['gda'] =  gda_IC.mean() + gda_IC.std() * 2, gda_IC.mean() - gda_IC.std() * 2
scores['ada'] =  ada_IC.mean() + ada_IC.std() * 2, ada_IC.mean() - ada_IC.std() * 2

#Cofidence interval 
scores
knn 0.7387387387387387
svm 0.7522522522522522
bayes 0.7477477477477478
lgc 0.7522522522522522
ada 0.7387387387387387
gda 0.7792792792792793
{'knn': (0.7791251472024496, 0.6427410628990546),
 'svm': (0.7492117252914164, 0.7431122729149348),
 'bayes': (0.7719999102027275, 0.7058933676905503),
 'lgc': (0.7492117252914164, 0.7431122729149348),
 'gda': (0.7583257028582367, 0.7213530709381842),
 'ada': (0.7565857437158992, 0.7194894264769183)}
#Avaliando o desempenho dos modelos que tiverem a acurácia dentro do intervalo
# para evitar o paradoxo da Acurácia

from sklearn.metrics import classification_report



models = {}

models['knn'] = KNeighborsClassifier()
models['svm'] = SVC()
models['bayes'] = GaussianNB()
models['lgc'] = LogisticRegression(max_iter=1000)
models['ada'] = AdaBoostClassifier()
models['gda'] = GradientBoostingClassifier()


for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    classification = classification_report(y_test,y_pred)
    print(name, classification)
knn               precision    recall  f1-score   support

           0       0.79      0.89      0.84       167
           1       0.46      0.29      0.36        55

    accuracy                           0.74       222
   macro avg       0.62      0.59      0.60       222
weighted avg       0.71      0.74      0.72       222

svm               precision    recall  f1-score   support

           0       0.75      1.00      0.86       167
           1       0.00      0.00      0.00        55

    accuracy                           0.75       222
   macro avg       0.38      0.50      0.43       222
weighted avg       0.57      0.75      0.65       222

bayes               precision    recall  f1-score   support

           0       0.76      0.98      0.85       167
           1       0.40      0.04      0.07        55

    accuracy                           0.75       222
   macro avg       0.58      0.51      0.46       222
weighted avg       0.67      0.75      0.66       222

lgc               precision    recall  f1-score   support

           0       0.75      1.00      0.86       167
           1       0.00      0.00      0.00        55

    accuracy                           0.75       222
   macro avg       0.38      0.50      0.43       222
weighted avg       0.57      0.75      0.65       222

ada               precision    recall  f1-score   support

           0       0.76      0.96      0.85       167
           1       0.36      0.07      0.12        55

    accuracy                           0.74       222
   macro avg       0.56      0.52      0.48       222
weighted avg       0.66      0.74      0.67       222

gda               precision    recall  f1-score   support

           0       0.80      0.94      0.87       167
           1       0.62      0.29      0.40        55

    accuracy                           0.78       222
   macro avg       0.71      0.62      0.63       222
weighted avg       0.76      0.78      0.75       222

Veja como a acurácia pode enganar a escolha do modelo. Os únicos que teveram a capacidade de classificar os indíviduos como possíveis inadimplentes (Recall), foram o KNN, com 18%, beysiano com 4%, Ada com 11%, e o Gda com 18%. Ou seja, do total de inadimplentes existentes na base proposta, apenas 4 modelos coseguiram fazer essa classificação. Todos os demais conseguiram prever apenas os não inadimplentes, que não é o objetivo de análise desse trabalho.
Qual a importância de analisar o Recall dos modelos nos estudos de crédito?
Quando uma empresa crediticia deseja fornecer crédito aos seus clientes, ela não só analisa as acurácias dos modelos. Na verdade isso componhe a menor parte na análise de crédito. Dado as condições internas da instituição, existe sempre um grau de risco nas aplicações de produtos financeiros, e de acordo com a situação interna da empresa, ela determinará qual individuo receberá seu crédito. Portanto, o ponto que mais afeta quem receberá o crédito, é saber qual é a probabilidade de um cliente com determinadas características vir a se tornar um possível inadimplente, e com isso saber qual é a sua probabilidade de ter atrasos, ou não quitação da dívida, e assim determinar o ponto de corte de acordo com o grau de risco que a instituição escolheu. Por exemplo, o banco X não aumentará o limite de cartão de crédito a clientes que possuem probabilidades maior ou igual a 30% de ser inadimplente. Isso equivale a determinar um ponto de corte de 0,3. Ou seja, nas decisões de quem receberá crédito ou não, não é a acurácia que nos traz o melhor desempenho do modelo, mas sim, o seu desempenho quanto a variações nos pontos de cortes, obtido pela AUC da curva ROC que é traçada a partir do Recall.
No nosso caso, o único modelo que teve a mínima capacidade classificar os positivos, foi o Beysiano. Vamos obter a AUC para comparar.
    from sklearn.metrics import roc_auc_score

    model_bayes = GaussianNB().fit(x_train, y_train)
    model_ada = AdaBoostClassifier().fit(x_train, y_train)
    model_knn = KNeighborsClassifier().fit(x_train, y_train)
    model_gda = GradientBoostingClassifier().fit(x_train, y_train)
    
  
    y_bayes = model_bayes.predict(x_test)
    y_ada = model_ada.predict(x_test)
    y_knn = model_knn.predict(x_test)
    y_gda = model_gda.predict(x_test)
    
    
auc = {}
auc['bayes'] = roc_auc_score(y_test, y_bayes)
auc['ada'] = roc_auc_score(y_test, y_ada)
auc['knn'] = roc_auc_score(y_test, y_knn)
auc['gda'] = roc_auc_score(y_test, y_gda)

auc
{'bayes': 0.5091997822536745,
 'ada': 0.515405552531301,
 'knn': 0.5885683179096354,
 'gda': 0.6155144256940664}
#Agora vamos interpretar nosso modelo com o Lime (ou Shap)
 
 
 