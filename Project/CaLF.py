from math import gamma, pi, sin
import numpy as np
from random import normalvariate, choice, randint
from SwarmPackagePy import intelligence
from numpy.random import random
import matplotlib.pyplot as plt


class CaLF(intelligence.sw):
    """
    Cat Swarm Optimization with Levy Flight
    """

    def __init__(self, n, function, lb, ub, dimension, iteration, mr=25, smp=10,
                 spc=False, cdc=1, srd=0.05):

        """
        :param n: number of agents
        :param function: test function
        :param lb: lower limits for plot axes
        :param ub: upper limits for plot axes
        :param dimension: space dimension
        :param iteration: number of iterations
        :param mr: number of cats that hunt (default value is 10)
        :param smp: seeking memory pool (default value is 2)
        :param spc: self-position considering (default value is False)
        :param cdc: counts of dimension to change (default value is 1)
        :param srd: seeking range of the selected dimension
        (default value is 0.1)
        """

        super(CaLF, self).__init__()

        self.__agents = np.random.uniform(lb, ub, (n, dimension))
        velocity = np.zeros((n, dimension))
        self._points(self.__agents)

        beta = 3 / 2
        sigma = (gamma(1 + beta) * sin(pi * beta / 2) / (
                gamma((1 + beta) / 2) * beta *
                2 ** ((beta - 1) / 2))) ** (1 / beta)
        u = np.array([normalvariate(0, 1) for k in range(dimension)]) * sigma
        v = np.array([normalvariate(0, 1) for k in range(dimension)])
        step = u / abs(v) ** (1 / beta)

        Pbest = self.__agents[np.array([function(x)
                                        for x in self.__agents]).argmin()]

        Gbest = Pbest

        flag = self.__set_flag(n, mr)
        if spc:
            sm = smp - 1
        else:
            sm = smp

        pltDataX = []
        pltDataY = []

        for t in range(iteration):

            for i in range(n):

                if flag[i] == 0:

                    if spc:

                        cop = self.__change_copy([self.__agents[i]], cdc, srd)[
                            0]
                        tmp = [self.__agents[i] for j in range(sm)]
                        tmp.append(cop)
                        copycat = np.array(tmp)

                    else:
                        copycat = np.array([self.__agents[i] for j in range(
                            sm)])
                    copycat = self.__change_copy(copycat, cdc, srd)

                    if copycat.all() == np.array(
                            [copycat[0] for j in range(sm)]).all():
                        P = np.array([1 for j in range(len(copycat))])

                    else:

                        fb = min([function(j) for j in copycat])
                        fmax = max([function(j) for j in copycat])
                        fmin = min([function(j) for j in copycat])
                        P = np.array(
                            [abs(function(j) - fb) / (fmax - fmin) for j in
                             copycat])
                    self.__agents[i] = copycat[P.argmax()]

                else:

                    for i in range(n):
                        stepsize = 0.2 * step * (self.__agents[i] - Pbest)
                        self.__agents[i] += stepsize * np.array([normalvariate(0, 1)
                                                                 for k in range(dimension)])

            Pbest = self.__agents[
                np.array([function(x) for x in self.__agents]).argmin()]

            pltDataX.append(t)
            pltDataY.append(function(Pbest))

            if function(Pbest) < function(Gbest):
                Gbest = Pbest
                #print(function(Gbest), t)
                self.__agents = np.clip(self.__agents, lb, ub)
            flag = self.__set_flag(n, mr)
            self._points(self.__agents)

        print(function(Pbest), t)
        plt.title('CSO With Levy Flight')
        plt.xlabel('Iterations')
        plt.ylabel('Normalised Cost Function')
        plt.plot(pltDataX, pltDataY)
        plt.show()

        self._set_Gbest(Gbest)

    def __set_flag(self, n, mr):

        flag = [0 for i in range(n)]
        m = mr
        while m > 0:
            tmp = randint(0, n - 1)
            if flag[tmp] == 0:
                flag[tmp] = 1
                m -= 1

        return flag

    def __change_copy(self, copycat, cdc, crd):

        for i in range(len(copycat)):
            flag = [0 for k in range(len(copycat[i]))]
            c = cdc
            while c > 0:
                tmp = randint(0, len(copycat[i]) - 1)
                if flag[tmp] == 0:
                    c -= 1
                    copycat[i][tmp] = copycat[i][tmp] + choice([-1, 1]) * crd

        return copycat



