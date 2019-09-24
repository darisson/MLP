from MLP import Rede
from ReaderManager import ReaderManager

# Ensinar a rede a reconhecer o padrao XOR

entradas = ReaderManager.get_entradas_xor()
respostas = ReaderManager.get_respostas_xor()
# entradas = ReaderManager.get_entradas()
# respostas = ReaderManager.get_respostas()

entradas_saidas = []

for index, e in enumerate(entradas):
    entradas_saidas.append([])
    entradas_saidas[index].append(entradas[index])
    entradas_saidas[index].append(respostas[index])

MOMENTUM = 0.5
TAXA_DE_APRENDIZADO = 0.1
PRECISAO = 0.0000001
MAX_ITERACOES = 10000
QTD_ENTRADAS = len(entradas_saidas[0][0])
QTD_NEURONIOS_CAMADA_ESCONDIDA = 2
QTD_SAIDAS = len(entradas_saidas[0][1])
ONLINE = True

print("MODE: {}".format("ONLINE" if ONLINE else "OFFLINE"))
print("TAXA_DE_APRENDIZADO: {}".format(TAXA_DE_APRENDIZADO))
print("PRECISAO: {}".format(PRECISAO))
print("MAX_ITERACOES: {}".format(MAX_ITERACOES))
print("QTD_ENTRADAS: {}".format(QTD_ENTRADAS))
print("QTD_NEURONIOS_CAMADA_ESCONDIDA: {}".format(QTD_NEURONIOS_CAMADA_ESCONDIDA))
print("QTD_SAIDAS: {}".format(QTD_SAIDAS))

rede = Rede(QTD_ENTRADAS, QTD_NEURONIOS_CAMADA_ESCONDIDA, QTD_SAIDAS, TAXA_DE_APRENDIZADO, MAX_ITERACOES, PRECISAO, MOMENTUM)

file = open("./Files/result{}.txt".format(1), "a")
file.write("MODE: {}\n".format("ONLINE" if ONLINE else "OFFLINE"))
file.write("TAXA_DE_APRENDIZADO: {}\n".format(TAXA_DE_APRENDIZADO))
file.write("PRECISAO: {}\n".format(PRECISAO))
file.write("MAX_ITERACOES: {}\n".format(MAX_ITERACOES))
file.write("QTD_ENTRADAS: {}\n".format(QTD_ENTRADAS))
file.write("QTD_NEURONIOS_CAMADA_ESCONDIDA: {}\n".format(QTD_NEURONIOS_CAMADA_ESCONDIDA))
file.write("QTD_SAIDAS: {}\n".format(QTD_SAIDAS))

if ONLINE:
    rede.treinar(entradas_saidas, file)
else:
    rede.treinar_offline(entradas_saidas)

rede.teste(entradas_saidas, respostas, file)

file.close()
