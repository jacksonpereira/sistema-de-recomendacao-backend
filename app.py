import time
from sklearn.cluster import AffinityPropagation
from sklearn.model_selection import ShuffleSplit
import math
import numpy as np
from flask import Flask, jsonify, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
print("Api disponível em 'http://localhost:5000'")

# Funções necessárias


def appendCluster(teste, arrayAgrup):
    cols = ['Nome', 'Plataforma', 'Ano de lancamento',
            'Genero', 'Classificacao', 'Agrupamento']
    resultado = pd.DataFrame(data={}, columns=cols)
    for x in arrayAgrup:
        resultado = resultado.append(
            teste[teste['Agrupamento'] == x], ignore_index=True)
    return resultado


def predicao(x, cluster):
    return cluster.predict([x])[0]


def doAggregation(gen, plat, dict):
    if "{}; {}".format(gen, plat) in dict:
        dict["{}; {}".format(gen, plat)] = dict["{}; {}".format(gen, plat)] + 1
    else:
        dict["{}; {}".format(gen, plat)] = 1


def getPureza(df):
    unicos = df['Agrupamento'].unique()
    dictPurify = {}
    arrPureza = []
    for x in unicos:
        dfAgrupamento = df[df['Agrupamento'] == x]
        arrLocalPurify = []
        dfAgrupamento.apply(lambda row: doAggregation(
            row["Plataforma"], row["Genero"], dictPurify), axis=1)
        for k, v in dictPurify.items():
            arrLocalPurify.append(
                dictPurify[k]/dfAgrupamento['Agrupamento'].size)
        arrPureza.append(max(arrLocalPurify))
        dictPurify.clear()
    return sum(arrPureza)/len(arrPureza)


def getEntropia(df):
    dictEntropy = {}
    arrEntropia = []
    entropia = 0
    unicos = df['Agrupamento'].unique()
    for x in unicos:
        dfAgrupamento = df[df['Agrupamento'] == x]
        arrLocalEntropy = []
        dfAgrupamento.apply(lambda row: doAggregation(
            row["Plataforma"], row["Genero"], dictEntropy), axis=1)
        for k, v in dictEntropy.items():
            arrLocalEntropy.append((dictEntropy[k]/dfAgrupamento['Agrupamento'].size)*math.log(
                dictEntropy[k]/dfAgrupamento['Agrupamento'].size))
        arrEntropia.append(sum(arrLocalEntropy)*(-1))
        for x in arrEntropia:
            entropia = entropia + \
                (x*(dfAgrupamento['Agrupamento'].size/df.size))
        dictEntropy.clear()
    return entropia


def separarDados(arrVersion):
    rs = ShuffleSplit(n_splits=arrVersion, test_size=0.33, random_state=42)
    dataRows = np.array(pd.read_csv('fileCleaned.csv').values)
    dataTreino = []
    dataTeste = []
    for train_index, test_index in rs.split(dataRows):
        # Treino
        for x in train_index:
            dataTreino.append(dataRows[x])
        # Teste
        for x in test_index:
            dataTeste.append(dataRows[x])
        salvarDados(dataTreino, dataTeste, arrVersion)
        arrVersion = arrVersion-1


def salvarDados(train, test, version):
    cols = fileCleanedColumns = pd.read_csv('fileCleaned.csv').columns
    pd.DataFrame(data=train, columns=cols).to_csv(
        r'dados/treino_{}.csv'.format(version), sep=',', index=False)
    pd.DataFrame(data=test, columns=cols).to_csv(
        r'dados/teste_{}.csv'.format(version), sep=',', index=False)


def treinamento(clf, dataTreino):
    step = 0
    for x in range(0, 11200, 500):
        if step <= 21:
            print('Step de treinamento: ', step)
            clf = clf.fit(np.array(dataTreino)[x:x+500])
        else:
            print('Step de treinamento: ', step)
            clf = clf.fit(np.array(dataTreino)[x:])
        step = step+1
    print("Step do treinamento: ", clf)
    return clf

def calcSimItem(x, y):
    item = 0
    if((x/y) < 1):
        item = x/y
    elif((x/y) > 1):
        item = (y/x)
    else:
        item = 1
    return item

def similaridade(x, option):
    x = [x['Plataforma'], x['Genero']]
    sim = 0
    for item in option:
        item1 = calcSimItem(x[0], item[0])
        item2 = calcSimItem(x[1], item[1])
        return (item1+item2)/2
    return sim


def salvarAgrupamento(row, agrup):
    db = pymysql.connect("localhost", "root", "arkannus34k",
                         "sistema_recomendacao")
    cursor = db.cursor()
    nome = "'" + str(row['Nome']).replace("'", "")+"'"
    cursor.execute(
        f"INSERT INTO predicoes (Nome, Plataforma, Ano_de_lancamento, Genero, Publicadora, Vendas_globais, Desenvolvedora, Classificacao, Media_de_pontuacao, Agrupamento) VALUES ({nome},{str(row['Plataforma'])},{str(row['Ano de lancamento'])},{str(row['Genero'])},{str(row['Publicadora'])},{str(row['Vendas globais'])},{str(row['Desenvolvedora'])},{str(row['Classificacao'])},{str(row['Media de pontuacao'])},{str(agrup)});")
    db.commit()
    db.close()


# Treinamento
print("Inicio do processamento: ", time.strftime("%H:%M:%S"))
p = 0
dataTreino = np.array(pd.read_csv(
    'dados/treino_sem_nulos.csv').loc[0:, ['Genero', 'Plataforma']].values)
clf = AffinityPropagation(affinity='euclidean', convergence_iter=15,
                          copy=False, damping=0.5, max_iter=200, preference=None, verbose=False)
timeStart = time.strftime("%H:%M:%S")
clf.fit(dataTreino)
clfGlobal = clf
cols = ['Nome', 'Plataforma', 'Ano de lancamento', 'Genero', 'Publicadora',
        'Vendas globais', 'Desenvolvedora', 'Classificacao', 'Media de pontuacao', 'Agrupamento']
teste = pd.read_csv('dados/teste_sem_nulos.csv')
teste['Agrupamento'] = teste.apply(lambda row: predicao(
    [row['Plataforma'], row['Genero']], clf), axis=1)
testeGlobal = teste
pd.DataFrame(data=teste, columns=cols).to_csv(
    r'dados/teste_sem_nulos.csv', sep=',', index=False)

timeEnd = time.strftime("%H:%M:%S")
# Validando modelo
pureza = getPureza(teste)
entropia = getEntropia(teste)
# Calculando resultados
colunas = ['Versão', 'Pureza', 'Entropia', 'Começo',
           'Término', 'Qtde Clusters', 'Parâmetros do cluster']
qtdeCluster = teste['Agrupamento'].nunique()
result = pd.DataFrame(data=[], columns=colunas)
result = result.append(pd.DataFrame(
    data=[[p, pureza, entropia, timeStart, timeEnd, qtdeCluster, str(clf.get_params())]], columns=colunas))
print("##########################################")
print("Resultado: ", result)
print("##########################################")
avaliacao = pd.DataFrame(
    data=[], columns={'Escolhas de jogos', 'Recomendações', 'Curtida'})
print("Término do processamento: ", time.strftime("%H:%M:%S"))

# Rotas
# Info da api
@app.route('/api/probe', methods=['GET'])
def probe():
    return result.to_json(orient='records'), 200

# Busca de jogos
@app.route('/search/<name>', methods=['GET'])
def search(name):
    lista = []
    for i, row in teste[teste['Nome'].str.contains(name)].iterrows():
        lista.append({'nome': row['Nome'], 'plataforma': row['Plataforma'], 'ano': row['Ano de lancamento'], 'genero': row['Genero'], 'publicadora': row['Publicadora'],'vendas': row['Vendas globais'], 'desenvolvedora': row['Desenvolvedora'], 'classificacao': row['Classificacao'], 'pontuacao': row['Media de pontuacao'], 'agrupamento': row['Agrupamento']})
    return jsonify(lista), 200

# Recomendação de jogos
@app.route('/recomendation', methods=['POST'])
def recomendation():
    data = request.get_json()
    if(len(data) == 0):
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ---- Size = 0 ----")
        resultado = teste.sort_values(by=['Vendas globais'], ascending=False)
        lista = []
        for i, row in resultado.iterrows():
            lista.append({'nome': row['Nome'], 'plataforma': row['Plataforma'], 'ano': row['Ano de lancamento'], 'genero': row['Genero'], 'publicadora': row['Publicadora'],
                        'vendas': row['Vendas globais'], 'desenvolvedora': row['Desenvolvedora'], 'classificacao': row['Classificacao'], 'pontuacao': row['Media de pontuacao'], 'agrupamento': row['Agrupamento']})
        return jsonify(lista), 200
    else:
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX ---- Size > 0 ----")
        # data = request.get_json()
        arrayAgrup = clf.predict(data)
        resultado = appendCluster(teste, arrayAgrup)
        resultado['Similaridade'] = resultado.apply(lambda row: similaridade(
            row, data), axis=1)
        resultado = resultado.sort_values(by=['Similaridade'], ascending=False)
        lista = []
        for i, row in resultado.iterrows():
            lista.append({'nome': row['Nome'], 'plataforma': row['Plataforma'], 'ano': row['Ano de lancamento'], 'genero': row['Genero'], 'publicadora': row['Publicadora'],
                        'vendas': row['Vendas globais'], 'desenvolvedora': row['Desenvolvedora'], 'classificacao': row['Classificacao'], 'pontuacao': row['Media de pontuacao'], 'agrupamento': row['Agrupamento'], 'similaridade': row['Similaridade']})
        return jsonify(lista), 200

# Avaliação da recomendação
@app.route('/recomendation/evaluation', methods=['POST'])
def evaluation():
    data = request.get_json()
    option = data['option']
    recomendation = data['recomendation']
    nota = data['nota']
    avaliacao.append(pd.DataFrame(data=[{str(option), str(recomendation), str(
        nota)}], columns={'Escolhas de jogos', 'Recomendações', 'Curtida'}))
    return jsonify(nota), 200


if __name__ == '__main__':
    app.run(debug=True)
