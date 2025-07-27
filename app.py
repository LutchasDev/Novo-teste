from flask import Flask, render_template, request
import pandas as pd
from sklearn.cluster import KMeans, MeanShift
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Carrega e prepara os dados
df = pd.read_excel('dados_ibge.xlsx', header=None, skiprows=6, nrows=3)

#print("DataFrame lido do Excel:")
#print(df)
#print("Colunas detectadas:", df.columns.tolist())

df.columns = ['Padrao', 'Residencial_3q', 'Residencial_4q']

#print("\nDataFrame após renomear colunas:")
#print(df)

df['Padrao_num'] = [0, 1, 2]  # Ajuste conforme a ordem dos seus dados

df_long = pd.melt(df, id_vars=['Padrao', 'Padrao_num'],
                  value_vars=['Residencial_3q', 'Residencial_4q'],
                  var_name='Tipo', value_name='Custo_m2')
df_long['Tipo_num'] = df_long['Tipo'].map({'Residencial_3q': 0, 'Residencial_4q': 1})

X = df_long[['Tipo_num', 'Padrao_num', 'Custo_m2']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Treinamento dos modelos de cluster
kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
meanshift = MeanShift().fit(X_scaled)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calcular', methods=['GET', 'POST'])
def calcular():
    
    if request.method == 'POST':
        metragem = float(request.form['metragem'])
        tipo = int(request.form['tipo'])       # 0 ou 1
        padrao = int(request.form['padrao'])   # 0, 1 ou 2

        # Obtem custo médio compatível com os inputs
        filtro = (df_long['Tipo_num'] == tipo) & (df_long['Padrao_num'] == padrao)
        custo_m2 = df_long.loc[filtro, 'Custo_m2'].mean()
        custo_total = metragem * custo_m2

        # Novo ponto para clustering
        novo_input = scaler.transform([[tipo, padrao, custo_m2]])
        cluster_kmeans = kmeans.predict(novo_input)[0]
        cluster_meanshift = meanshift.predict(novo_input)[0]

        return render_template('calcular.html', 
                            metragem=metragem,
                            custo_m2=custo_m2,
                            custo_total=custo_total,
                            cluster_kmeans=cluster_kmeans,
                            cluster_meanshift=cluster_meanshift
        )
    else:
        return f"Recebido via {request.method}", 200

if __name__ == '__main__':
    app.run(debug=True)
