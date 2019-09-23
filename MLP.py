

import math
import random
import numpy

# random.seed(1)
# General Functions
def rand(a, b):

    return random.uniform(a, b)

def funcao_ativacao_tangente_hiperbolica(x): # funcao tangente hiperbolica
    return math.tanh(x)

def derivada_funcao_ativacao(x): # derivada da tangente hiperbolica
    return 1/(math.cosh(x)**2)

class Rede:

    def __init__(self, qtd_entrada, qtd_neuronios_camada_escondida, qtd_neuronios_camada_saida, taxa_aprendizado, max_interacoes, precisao):

        self.taxa_aprendizado = taxa_aprendizado
        self.max_interacoes = max_interacoes
        self.precisao = precisao

        self.qtd_entrada = qtd_entrada # +1 -> bias
        self.qtd_neuronios_camada_escondida = qtd_neuronios_camada_escondida # +1 -> bias
        self.qtd_neuronios_camada_saida = qtd_neuronios_camada_saida

        self.I_camada_escondida = numpy.ones(self.qtd_neuronios_camada_escondida)
        self.I_camada_de_saida = numpy.ones(self.qtd_neuronios_camada_saida)

        self.entrada = numpy.ones(self.qtd_entrada)
        self.saida_camada_escondida = numpy.ones(self.qtd_neuronios_camada_escondida)
        self.saida_camada_de_saida = numpy.ones(self.qtd_neuronios_camada_saida)

        # self.entrada[self.qtd_entrada - 1] = -1 # ENTRADA BIAS CAMADA ESCONDIDA
        # self.saida_camada_escondida[self.qtd_neuronios_camada_escondida - 1] = -1 # ENTRADA BIAS CAMADA DE SAIDA

        self.setup_pesos_camada_escondida()
        self.setup_pesos_camada_de_saida()
        print("---------------------------------------------------------------------------")


    def setup_pesos_camada_escondida(self):
        self.pesos_camada_escondida = numpy.ones((self.qtd_entrada + 1, self.qtd_neuronios_camada_escondida))

        print("CAMADA ESCONDIDA: ---------------------------------------------------------")
        for i in range(self.qtd_entrada + 1): # Camada escondida
            for j in range(self.qtd_neuronios_camada_escondida):
                self.pesos_camada_escondida[i][j] = rand(-0.5, 0.5)
            print("Peso[{}]: {}".format(i, self.pesos_camada_escondida[i]))


    def setup_pesos_camada_de_saida(self):

        self.pesos_camada_saida = numpy.ones((self.qtd_neuronios_camada_escondida + 1, self.qtd_neuronios_camada_saida))

        print("CAMADA DE SAIDA: ---------------------------------------------------------")
        for i in range(self.qtd_neuronios_camada_escondida + 1): # Camada de saida
            for j in range(self.qtd_neuronios_camada_saida):
                self.pesos_camada_saida[i][j] = rand(-0.5, 0.5)
            print("Peso[{}]: {}".format(i, self.pesos_camada_saida[i]))

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
                somatorio = somatorio + self.entrada[i] * self.pesos_camada_escondida[i][j]
            somatorio = somatorio + (-1 * self.pesos_camada_escondida[self.qtd_entrada][j])
            self.I_camada_escondida[j] = somatorio
            self.saida_camada_escondida[j] = funcao_ativacao_tangente_hiperbolica(somatorio)

    def run_camada_de_saida(self):
        for j in range(self.qtd_neuronios_camada_saida):
            somatorio = 0
            for i in range(self.qtd_neuronios_camada_escondida):
                somatorio = somatorio + self.saida_camada_escondida[i] * self.pesos_camada_saida[i][j]
            somatorio = somatorio + (-1 * self.pesos_camada_saida[self.qtd_neuronios_camada_escondida][j])
            self.I_camada_de_saida[j] = somatorio
            self.saida_camada_de_saida[j] = funcao_ativacao_tangente_hiperbolica(somatorio)

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
            erro = erro + ((saidas_desejadas[i] - self.saida_camada_de_saida[i]) ** 2)
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

        for i in range(self.qtd_neuronios_camada_escondida):
            for j in range(self.qtd_neuronios_camada_saida):
                change = residuos_saida[j] * self.saida_camada_escondida[i]
                self.pesos_camada_saida[i][j] = self.pesos_camada_saida[i][
                    j] + (self.taxa_aprendizado * change)

    def ajustar_pesos_camada_escondida(self, residuos_escondida):

        for i in range(self.qtd_entrada):
            for j in range(self.qtd_neuronios_camada_escondida):
                change = residuos_escondida[j] * self.entrada[i]
                self.pesos_camada_escondida[i][j] = self.pesos_camada_escondida[i][
                    j] + (self.taxa_aprendizado * change)

    def teste(self, entradas_saidas):
        for p in entradas_saidas:
            array = self.forward(p[0])
            print("Entradas: " + str(p[0]) + ' - Saída encontrada: ' + str(array[0]))

    def treinar(self, entradas_saidas):

        epocas = 0

        eqm = 1000
        erro_anterior = 0

        while (epocas < self.max_interacoes and abs(eqm - erro_anterior) > 0.01):

            eqm = 0
            erro_anterior = eqm
            erro = 0
            for p in entradas_saidas:

                entradas = p[0]
                saidas_desejadas = p[1]

                self.forward(entradas)
                erro = erro + self.backward(saidas_desejadas)
            eqm = erro/len(entradas_saidas)
            if epocas % 100 == 0:
                print("Treinando epoca: {}".format(epocas))

            epocas += 1

        print("Epocas: {}".format(epocas))
