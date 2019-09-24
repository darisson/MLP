

import math
import random
import numpy
from sklearn.utils import shuffle

random.seed(2)
# General Functions
def rand(a, b):
    return random.uniform(a, b)

def funcao_ativacao_tangente_hiperbolica(x): # funcao tangente hiperbolica
    return math.tanh(x)

def derivada_funcao_ativacao(x): # derivada da tangente hiperbolica
    return 1/(math.cosh(x)**2)

class Rede:

    def __init__(self, qtd_entrada, qtd_neuronios_camada_escondida, qtd_neuronios_camada_saida, taxa_aprendizado, max_interacoes, precisao, momentum):

        self.taxa_aprendizado = taxa_aprendizado
        self.max_interacoes = max_interacoes
        self.precisao = precisao
        self.momentum = momentum

        self.qtd_entrada = qtd_entrada # +1 -> bias
        self.qtd_neuronios_camada_escondida = qtd_neuronios_camada_escondida # +1 -> bias
        self.qtd_neuronios_camada_saida = qtd_neuronios_camada_saida

        # Init Online ------------------------------------------------------
        self.I_camada_escondida = numpy.ones(self.qtd_neuronios_camada_escondida)
        self.I_camada_de_saida = numpy.ones(self.qtd_neuronios_camada_saida)

        self.entrada = numpy.ones(self.qtd_entrada)
        self.saida_camada_escondida = numpy.ones(self.qtd_neuronios_camada_escondida)
        self.saida_camada_de_saida = numpy.ones(self.qtd_neuronios_camada_saida)
        # -------------------------------------------------------------------

        # Init Offline ------------------------------------------------------
        self.I_camada_escondida_off = []
        self.I_camada_de_saida_off = []

        self.saida_camada_escondida_off = []
        self.saida_camada_saida_off = []
        # -------------------------------------------------------------------

        # self.entrada[self.qtd_entrada - 1] = -1 # ENTRADA BIAS CAMADA ESCONDIDA
        # self.saida_camada_escondida[self.qtd_neuronios_camada_escondida - 1] = -1 # ENTRADA BIAS CAMADA DE SAIDA

        self.setup_pesos_camada_escondida()
        self.setup_pesos_camada_de_saida()
        print("---------------------------------------------------------------------------")


    def setup_pesos_camada_escondida(self):
        self.pesos_camada_escondida = numpy.ones((self.qtd_entrada + 1, self.qtd_neuronios_camada_escondida))

        print("CAMADA ESCONDIDA: ---------------------------------------------------------")
        for i in range(self.qtd_entrada + 1): # Camada escondida

            print("Peso[{}]:".format(i))

            for j in range(self.qtd_neuronios_camada_escondida):
                self.pesos_camada_escondida[i][j] = rand(-0.5, 0.5)
                print("{}".format(self.pesos_camada_escondida[i][j]))


    def setup_pesos_camada_de_saida(self):

        self.pesos_camada_saida = numpy.ones((self.qtd_neuronios_camada_escondida + 1, self.qtd_neuronios_camada_saida))

        print("CAMADA DE SAIDA: ---------------------------------------------------------")

        for i in range(self.qtd_neuronios_camada_escondida + 1): # Camada de saida

            print("Peso[{}]:".format(i))

            for j in range(self.qtd_neuronios_camada_saida):
                self.pesos_camada_saida[i][j] = rand(-0.5, 0.5)
                print("{}".format(self.pesos_camada_saida[i][j]))

    def treinar(self, entradas_saidas, file, embaralhar):

        epocas = 0

        eqm = 0
        erro_anterior = 1000
        while (epocas < self.max_interacoes and abs(eqm - erro_anterior) > self.precisao):

            if embaralhar:
                entradas_saidas = shuffle(entradas_saidas, random_state=0)

            erro_anterior = eqm
            erro = 0
            for p in entradas_saidas:
                entradas = p[0]
                saidas_desejadas = p[1]

                self.forward(entradas)
                erro += self.backward(saidas_desejadas)
            eqm = erro / len(entradas_saidas)
            # if epocas % 100 == 0:
            print("Treinando epoca: {} / erro: {}".format(epocas, eqm - erro_anterior))

            epocas += 1

        print("Epocas: {}".format(epocas))
        file.write("Epocas: {}\n".format(epocas))
        # print("EQM: {} / EQM ANTERIOR: {}".format(eqm, erro_anterior))

    def teste(self, entradas_saidas, saida_desejada, file):
        # print("--------------------------------------")
        # print("Saida desejada:")

        # for index, saida in enumerate(saida_desejada):
        #     print("{}: {}".format(index, saida))

        # print("Saída encontrada:")
        qtd_acertos = 0
        for index, p in enumerate(entradas_saidas):
            array = self.forward(p[0])
            erro = False
            for i in range(len(array)):
                array[i] = round(array[i], 1)
                if ((array[i] < 0.8 and saida_desejada[index][i] == 1) or (
                        array[i] > -0.8 and saida_desejada[index][i] == -1)) and (not erro):
                    erro += True
            if not erro:
                qtd_acertos += 1
            # print("{}: {}".format(index, array))
        print("{}%".format(int(qtd_acertos / len(saida_desejada) * 100)))
        file.write("{}%".format(int(qtd_acertos / len(saida_desejada) * 100)))

        return int(qtd_acertos / len(saida_desejada) * 100)

    # FORWARD FUNCTIONS
    def forward(self, entradas):
        for i in range(self.qtd_entrada):
            self.entrada[i] = entradas[i]

        self.run_camada_escondida()
        self.run_camada_de_saida()

        return self.saida_camada_de_saida

    def run_camada_escondida(self):
        for j in range(self.qtd_neuronios_camada_escondida):
            somatorio = 0
            for i in range(self.qtd_entrada):
                somatorio += self.entrada[i] * self.pesos_camada_escondida[i][j]
            somatorio += (-1 * self.pesos_camada_escondida[self.qtd_entrada][j])

            self.I_camada_escondida[j] = somatorio

            self.saida_camada_escondida[j] = funcao_ativacao_tangente_hiperbolica(self.I_camada_escondida[j])

        self.I_camada_escondida_off.append(self.I_camada_escondida)
        self.saida_camada_escondida_off.append(self.saida_camada_escondida)

    def run_camada_de_saida(self):
        for j in range(self.qtd_neuronios_camada_saida):
            somatorio = 0
            for i in range(self.qtd_neuronios_camada_escondida):
                somatorio += self.saida_camada_escondida[i] * self.pesos_camada_saida[i][j]
            somatorio += (-1 * self.pesos_camada_saida[self.qtd_neuronios_camada_escondida][j])

            self.I_camada_de_saida[j] = somatorio

            self.saida_camada_de_saida[j] = funcao_ativacao_tangente_hiperbolica(self.I_camada_de_saida[j])

        self.I_camada_de_saida_off.append(self.I_camada_de_saida)
        self.saida_camada_saida_off.append(self.saida_camada_de_saida)

    # BACKWARD FUNCTIONS
    def backward(self, saidas_desejadas):
        # CAMADA DE SAIDA - RESIDUO
        residuos_saida = self.get_residuos_camada_saida(saidas_desejadas)

        # CAMADA DE ESCONDIDA - RESIDUO
        residuos_escondida = self.get_residuos_camada_escondida(residuos_saida)

        # CAMADA DE SAIDA - AJUSTE
        self.ajustar_pesos_camada_saida(residuos_saida)

        # CAMADA DE ESCONDIDA - AJUSTE
        self.ajustar_pesos_camada_escondida(residuos_escondida)

        # calculando erro
        erro = 0
        for i in range(len(saidas_desejadas)):
            erro += ((saidas_desejadas[i] - self.saida_camada_de_saida[i]) ** 2)/2
        return erro

    def get_residuos_camada_saida(self, respostas):

        residuos_saida = numpy.zeros(self.qtd_neuronios_camada_saida)
        erro = 0

        for i in range(self.qtd_neuronios_camada_saida):
            erro = respostas[i] - self.saida_camada_de_saida[i]
            residuos_saida[i] = derivada_funcao_ativacao(self.I_camada_de_saida[i]) * erro

        return residuos_saida

    def get_residuos_camada_escondida(self, residuos_saida):
        residuos_escondida = numpy.zeros(self.qtd_neuronios_camada_escondida)

        for i in range(self.qtd_neuronios_camada_escondida):
            erro = 0
            for j in range(self.qtd_neuronios_camada_saida):
                erro = erro + residuos_saida[j] * self.pesos_camada_saida[i][j]
            residuos_escondida[i] = derivada_funcao_ativacao(self.I_camada_escondida[i]) * erro

        return residuos_escondida

    def ajustar_pesos_camada_saida(self, residuos_saida):

        for i in range(self.qtd_neuronios_camada_escondida + 1):
            for j in range(self.qtd_neuronios_camada_saida):
                if i != self.qtd_neuronios_camada_escondida:
                    self.pesos_camada_saida[i][j] += (self.taxa_aprendizado * residuos_saida[j] * self.saida_camada_escondida[i])
                else:
                    self.pesos_camada_saida[i][j] += (self.taxa_aprendizado * residuos_saida[j] * -1)


    def ajustar_pesos_camada_escondida(self, residuos_escondida):

        for i in range(self.qtd_entrada + 1):
            for j in range(self.qtd_neuronios_camada_escondida):
                if i != self.qtd_entrada:
                    self.pesos_camada_escondida[i][j] += (self.taxa_aprendizado * residuos_escondida[j] * self.entrada[i])
                else:
                    self.pesos_camada_escondida[i][j] += (self.taxa_aprendizado * residuos_escondida[j] * -1)
# ------------------------------------------ Offline -------------------------------------------
    def treinar_offline(self, entradas_saidas):

        epocas = 0

        eqm = 0
        erro_anterior = 1000

        while (epocas < self.max_interacoes and abs(eqm - erro_anterior) > self.precisao):


            erro_anterior = eqm
            erro = 0
            self.saida_camada_escondida_off = []
            self.saida_camada_saida_off = []
            for amostra in entradas_saidas:
                entradas = amostra[0]
                self.forward(entradas)

            for k, amostra in enumerate(entradas_saidas):
                entradas = amostra[0]
                saidas_desejadas = amostra[1]

                IE = self.I_camada_escondida_off[k]
                YE = self.saida_camada_escondida_off[k]

                IS = self.I_camada_de_saida_off[k]
                YS = self.saida_camada_saida_off[k]

                erro += self.backward_offline(saidas_desejadas, IE, YE, IS, YS, entradas)
            eqm = erro/len(entradas_saidas)
            # if epocas % 100 == 0:
            print("Treinando epoca: {} / erro: {}".format(epocas, eqm))

            epocas += 1

        print("Epocas: {}".format(epocas))


    def backward_offline(self, saidas_desejadas, IE, YE, IS, YS, ent):
        # CAMADA DE SAIDA - RESIDUO
        residuos_saida = self.get_residuos_camada_saida_off(saidas_desejadas, IS, YS)

        # CAMADA DE ESCONDIDA - RESIDUO
        residuos_escondida = self.get_residuos_camada_escondida_off(residuos_saida, IE)

        # CAMADA DE SAIDA - AJUSTE
        self.ajustar_pesos_camada_saida_off(residuos_saida, YE)

        # CAMADA DE ESCONDIDA - AJUSTE
        self.ajustar_pesos_camada_escondida_off(residuos_escondida, ent)

        # calculando erro
        erro = 0
        for i in range(len(saidas_desejadas)):
            erro += ((saidas_desejadas[i] - self.saida_camada_de_saida[i]) ** 2)/2
        return erro

    def get_residuos_camada_saida_off(self, respostas, I, Y):

        residuos_saida = numpy.zeros(self.qtd_neuronios_camada_saida)
        erro = 0

        for i in range(self.qtd_neuronios_camada_saida):
            erro = respostas[i] - Y[i]
            residuos_saida[i] = derivada_funcao_ativacao(I[i]) * erro

        return residuos_saida

    def get_residuos_camada_escondida_off(self, residuos_saida, I):
        residuos_escondida = numpy.zeros(self.qtd_neuronios_camada_escondida)

        for i in range(self.qtd_neuronios_camada_escondida):
            erro = 0
            for j in range(self.qtd_neuronios_camada_saida):
                erro += residuos_saida[j] * self.pesos_camada_saida[i][j]
            residuos_escondida[i] = derivada_funcao_ativacao(I[i]) * erro

        return residuos_escondida

    def ajustar_pesos_camada_saida_off(self, residuos_saida, Y):

        for i in range(self.qtd_neuronios_camada_escondida):
            for j in range(self.qtd_neuronios_camada_saida):
                if i != self.qtd_entrada:
                    self.pesos_camada_saida[i][j] += (self.taxa_aprendizado * residuos_saida[j] * Y[i])
                else:
                    self.pesos_camada_escondida[i][j] += (self.taxa_aprendizado * residuos_saida[j] * -1)


    def ajustar_pesos_camada_escondida_off(self, residuos_escondida, ent):

        for i in range(self.qtd_entrada + 1):
            for j in range(self.qtd_neuronios_camada_escondida):
                if i != self.qtd_entrada:
                    self.pesos_camada_escondida[i][j] += (self.taxa_aprendizado * residuos_escondida[j] * ent[i])
                else:
                    self.pesos_camada_escondida[i][j] += (self.taxa_aprendizado * residuos_escondida[j] * -1)

