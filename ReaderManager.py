import math
import numpy as np

XOR_X = './Files/Xor.txt'

XOR_D = './Files/XorD.txt'

ENTRADAS = './Files/Xlarge.txt'

RESPOSTAS = './Files/Xsmall.txt'

TESTE_2 = './Files/Xnoise2.txt'
TESTE_5 = './Files/Xnoise5.txt'
TESTE_10 = './Files/Xnoise10.txt'


class ReaderManager:

    @staticmethod
    def get_entradas():
        file = ENTRADAS
        entradas = []
        with open(file) as f:
            conteudo = f.readlines()
        lines = [line.strip() for line in conteudo]

        for i in range(len(lines)):
            entradas.append([])
            vetor = lines[i].split()
            for pos in vetor:
                entradas[i].append(float(pos))

        return entradas

    @staticmethod
    def get_respostas():
        file = RESPOSTAS
        respostas = []
        with open(file) as f:
            conteudo = f.readlines()
        lines = [line.strip() for line in conteudo]

        for i in range(len(lines)):
            respostas.append([])
            vetor = lines[i].split()
            for pos in vetor:
                respostas[i].append(float(pos))

        return respostas

    @staticmethod
    def get_entradas_teste(num):
        if num == 2:
            file = TESTE_2
        elif num == 5:
            file = TESTE_5
        else:
            file = TESTE_10

        entradas = []
        with open(file) as f:
            conteudo = f.readlines()
        lines = [line.strip() for line in conteudo]

        for i in range(len(lines)):
            entradas.append([])
            vetor = lines[i].split()
            for pos in vetor:
                entradas[i].append(float(pos))

        return entradas

    @staticmethod
    def get_entradas_xor():
        file = XOR_X
        entradas = []
        with open(file) as f:
            conteudo = f.readlines()
        lines = [line.strip() for line in conteudo]

        for i in range(len(lines)):
            entradas.append([])
            vetor = lines[i].split()
            for pos in vetor:
                entradas[i].append(float(pos))

        return entradas

    @staticmethod
    def get_respostas_xor():
        file = XOR_D
        respostas = []
        with open(file) as f:
            conteudo = f.readlines()
        lines = [line.strip() for line in conteudo]

        for i in range(len(lines)):
            respostas.append([])
            vetor = lines[i].split()
            for pos in vetor:
                respostas[i].append(float(pos))

        return respostas

    @staticmethod
    def modulo(entradas):

        modulos = []
        for i in range(len(entradas)):
            soma = 0
            for x in entradas:
                soma += x[i]**2

            modulos.append(math.sqrt(soma))
        return modulos

    @staticmethod
    def normalizar(entradas):
        modulos = ReaderManager.modulo(entradas)
        entradasAux = []
        for i in range(len(entradas)):
            vetor = []
            for x in entradas:
                vetor.append(x[i]/modulos[i])
            entradasAux.append(vetor)

        return entradasAux
