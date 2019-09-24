from MLP import Rede
from ReaderManager import ReaderManager
import random
# Ensinar a rede a reconhecer o padrao XOR

# entradas = ReaderManager.get_entradas_xor()
# respostas = ReaderManager.get_respostas_xor()
entradas = ReaderManager.get_entradas()
entradasXnoise2 = ReaderManager.get_entradas_teste(2)
entradasXnoise5 = ReaderManager.get_entradas_teste(5)
entradasXnoise10 = ReaderManager.get_entradas_teste(10)
respostas = ReaderManager.get_respostas()

entradas_saidas = []
entradas_saidas_2 = []
entradas_saidas_5 = []
entradas_saidas_10 = []

for index, e in enumerate(entradas):
    entradas_saidas.append([])
    entradas_saidas[index].append(entradas[index])
    entradas_saidas[index].append(respostas[index])

for index, e in enumerate(entradasXnoise2):
    entradas_saidas_2.append([])
    entradas_saidas_2[index].append(entradasXnoise2[index])
    entradas_saidas_2[index].append(respostas[index])

for index, e in enumerate(entradasXnoise5):
    entradas_saidas_5.append([])
    entradas_saidas_5[index].append(entradasXnoise5[index])
    entradas_saidas_5[index].append(respostas[index])

for index, e in enumerate(entradas_saidas_10):
    entradas_saidas_10.append([])
    entradas_saidas_10[index].append(entradas_saidas_10[index])
    entradas_saidas_10[index].append(respostas[index])
# ---------------------------------------- CONFIGURACAO ------------------------------------------
MOMENTUM = 0.5
TAXA_DE_APRENDIZADO = 0.1
PRECISAO = 0.000001
MAX_ITERACOES = 10000
QTD_ENTRADAS = len(entradas_saidas[0][0])
QTD_NEURONIOS_CAMADA_ESCONDIDA = 10
QTD_SAIDAS = len(entradas_saidas[0][1])
EMBARALHAR = False
ONLINE = True
# ------------------------------------------------------------------------------------------------

print("\nMODE: {}".format("ONLINE" if ONLINE else "OFFLINE"))
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
file.write("QTD_SAIDAS: {}\n\n".format(QTD_SAIDAS))

# ---------------------------------------- TREINAMENTO --------------------------------------------
if ONLINE:
    rede.treinar(entradas_saidas, file, EMBARALHAR)
else:
    rede.treinar_offline(entradas_saidas)

# Teste com amostras de treinamento
file.write("Teste amostra de treinamento:  ")
p = rede.teste(entradas_saidas, respostas, file)
# ------------------------------------------------------------------------------------------------
# ------------------------------------------- TESTE ----------------------------------------------
file.write("\nTestes -------------------------------\n")
file.write("\nTeste Xnoise2:  ")
print("\nTeste Xnoise2:  ")
# Teste com amostras Xnoise2
P1 = rede.teste(entradas_saidas_2, respostas, file)

file.write("\nTeste Xnoise5:  ")
print("\nTeste Xnoise5:  ")
# Teste com amostras Xnoise5
P2 = rede.teste(entradas_saidas_5, respostas, file)

file.write("\nTeste Xnoise10:  ")
print("\nTeste Xnoise10:  ")
# Teste com amostras Xnoise10
P3 = rede.teste(entradas_saidas_10, respostas, file)

file.write("\nMedia taxa de recuperação: {}%\n".format((P1 + P2 + P3)/3))
# ------------------------------------------------------------------------------------------------

file.write("\n-------------------------------------------------------")
file.close()
