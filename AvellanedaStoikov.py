import math
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import matplotlib.pyplot as plt
import random

random.seed(100)


class AvellanedaStoikov:
    """
    Our aim in this class is to implement the Avellaneda-Stoikov model.
    Notations:
    - s(t): fair price process of the underlying stock
    - x(t): cash process
    - q(t): stock inventory (how many stocks the market maker is holding)

    In this case, we assume that the price moves in the dynamics of a Brownian motion.
    Stock inventory moves in Poisson process.
    """

    def __init__(self, S0, T, model: str):
        """

        :param S0: initial stock price
        :param q: stock inventory
        :param sigma: volatility of the underlying
        :param T: maturity of the simulation
        :param gamma: risk aversion coefficient
        :param k: liquidity value
        """

        # Variable models
        self.S0 = S0
        self.T = T
        self.model = model

        # Fixed models
        self.A = 0.05
        self.sigma = S0 * 0.02 / math.sqrt(T)
        self.k = math.log(2) / 0.01
        self.q_tilde = 100
        self.gamma = 0.01 / self.q_tilde
        self.n_steps = int(T)
        self.n_paths = 500
        self.h = 15 * 60
        self.adverse = False

        if model == 'AS':
            self.deltaBidZero = 0.5 * self.gamma * self.sigma ** 2 * self.T + 1 / self.gamma * math.log(
                1 + self.gamma / self.k) + self.gamma * self.sigma ** 2 * self.T * 0
            self.deltaAskZero = 0.5 * self.gamma * self.sigma ** 2 * self.T + 1 / self.gamma * math.log(
                1 + self.gamma / self.k) - self.gamma * self.sigma ** 2 * self.T * 0
        else:
            self.trivial_value = 0
            self.deltaAskZero = 0
            self.deltaBidZero = 0

        self.prices = None
        self.orders = None
        self.cashflow = None
        self.pnl = None
        self.bidPrice = None
        self.askPrice = None
        self.jump = 1

    def setTrivialDeltaValue(self, value):
        """
        Set value of the trivial model
        :return: None
        """
        self.trivial_value = value

        # Call the setTrivialDelta method to set the value
        self.setTrivialDelta(delta=value)

    def setTrivialDelta(self, delta):
        """
        Method that sets the initial values of trivial deltas using the trivial value attribute
        :return:
        """

        self.deltaBidZero = delta
        self.deltaAskZero = delta

    def setJump(self, jump=0.0):
        """
        Method that sets the jump for the Question 5. Introduced in the middle of the trading day.
        :param jump: float.
        :return: None
        """

        self.jump = 1 + jump

    def activateAdverseSelection(self, activate=False):
        """
        Method that activates adverse selection
        :return: None.
        """

        self.adverse = activate

    @staticmethod
    def sigmoid(x):
        """
        Method that returns the sigmoid value.
        :param x: float.
        :return: float. Sigmoid value of given x.
        """

        return 1 / (1 + math.exp(-x))

    def execute(self):
        """
        Method to return simulated price paths of the price
        :return: -
        """
        prices = np.zeros(self.n_steps)
        prices[0] = self.S0

        orders = np.zeros(self.n_steps)
        orders[0] = 0

        cashflow = np.zeros(self.n_steps)
        cashflow[0] = 0

        # Create the delta bid and asks and add the first value
        deltaBid = np.zeros(self.n_steps)
        deltaAsk = np.zeros(self.n_steps)
        deltaBid[0] = self.deltaBidZero
        deltaAsk[0] = self.deltaAskZero

        lambdaBid = np.zeros(self.n_steps)
        lambdaAsk = np.zeros(self.n_steps)
        lambdaBid[0] = self.A * math.exp(- self.k * self.deltaBidZero)
        lambdaAsk[0] = self.A * math.exp(- self.k * self.deltaAskZero)

        poissonBid = np.zeros(self.n_steps)
        poissonAsk = np.zeros(self.n_steps)
        poissonBid[0] = poisson.rvs(lambdaBid[0])
        poissonAsk[0] = poisson.rvs(lambdaAsk[0])

        pnl = np.zeros(self.n_steps)
        pnl[0] = cashflow[0] + orders[0] * prices[0]

        for t in range(1, self.n_steps):

            if t == self.n_steps // 2:
                # Calculate the Innovation step
                prices[t] = self.jump * prices[t - 1]
            else:
                # Calculate the Innovation step
                prices[t] = prices[t - 1] + np.random.normal(loc=0,
                                                             scale=1) * self.sigma

        for t in range(1, self.n_steps):

            # If the model we are using the AS model then we set the lambda and poisson process values
            if self.model == 'AS':
                deltaBid[t] = max(0.5 * self.gamma * self.sigma ** 2 * self.T + 1 / self.gamma * math.log(
                    1 + self.gamma / self.k) + self.gamma * self.sigma ** 2 * self.T * orders[t - 1], 0)
                deltaAsk[t] = max(0.5 * self.gamma * self.sigma ** 2 * self.T + 1 / self.gamma * math.log(
                    1 + self.gamma / self.k) - self.gamma * self.sigma ** 2 * self.T * orders[t - 1], 0)

            # Else we proceed with the initial delta values
            else:
                deltaBid[t] = self.deltaBidZero
                deltaAsk[t] = self.deltaAskZero

            # If adverse model is activated
            if self.adverse:
                if t + self.h >= self.n_steps:
                    lambdaBid[t] = self.A * math.exp(- self.k * deltaBid[t]) * (
                                1 - self.sigmoid(0.5 * (prices[-1] - prices[t]) / (self.sigma * math.sqrt(self.h))))
                    lambdaAsk[t] = self.A * math.exp(- self.k * deltaAsk[t]) * self.sigmoid(
                        0.5 * (prices[-1] - prices[t]) / (self.sigma * math.sqrt(self.h)))
                else:
                    lambdaBid[t] = self.A * math.exp(- self.k * deltaBid[t]) * (1 - self.sigmoid(
                        0.5 * (prices[t + self.h] - prices[t]) / (self.sigma * math.sqrt(self.h))))
                    lambdaAsk[t] = self.A * math.exp(- self.k * deltaAsk[t]) * self.sigmoid(
                        0.5 * (prices[t + self.h] - prices[t]) / (self.sigma * math.sqrt(self.h)))

            # If no adverse is being used
            else:
                lambdaBid[t] = self.A * math.exp(- self.k * deltaBid[t])
                lambdaAsk[t] = self.A * math.exp(- self.k * deltaAsk[t])

            uniCriteriaBid = np.random.uniform(0, 1)
            uniCriteriaAsk = np.random.uniform(0, 1)

            if uniCriteriaBid < lambdaBid[t]:
                poissonBid[t] = 1
            else:
                poissonBid[t] = 0

            if uniCriteriaAsk < lambdaAsk[t]:
                poissonAsk[t] = 1
            else:
                poissonAsk[t] = 0

            orders[t] = self.q_tilde * (poissonBid[t] - poissonAsk[t]) + orders[t - 1]

            cashflow[t] = self.q_tilde * (
                    (prices[t] + deltaAsk[t]) * poissonAsk[t] - (prices[t] - deltaBid[t]) * poissonBid[t]) + cashflow[
                              t - 1]
            pnl[t] = cashflow[t] + orders[t] * prices[t]

        self.prices = prices
        self.orders = orders
        self.cashflow = cashflow
        self.pnl = pnl
        self.bidPrice = prices - deltaBid
        self.askPrice = prices + deltaAsk

        return None

    def getPrices(self):
        """
        Method that returns prices.
        :return: list. self.prices
        """

        return self.prices

    def getOrders(self):
        """
        Method that returns orders.
        :return: list. self.orders
        """

        return self.orders

    def getCashflow(self):
        """
        Method that returns cashflow.
        :return: list. self.cashflow
        """

        return self.cashflow

    def getPnL(self):
        """
        Method that returns PnL.
        :return: list. self.pnl
        """

        return self.pnl

    def getFinalPnL(self):
        """
        Method that returns PnL.
        :return: double. self.pnl
        """

        return self.pnl[-1]

    def visualize(self):
        """
        Method to visualize market
        :return:
        """
        time_steps = list(range(self.n_steps))
        sns.color_palette("hls", 8)
        fig, ax = plt.subplots(4, 1,
                               gridspec_kw={'height_ratios': [3, 1, 1, 3]},
                               sharex=True,
                               figsize=(20, 10))

        ################### Plot values ###################
        g1_1 = sns.lineplot(x=time_steps,
                            y=self.prices,
                            ax=ax[0],
                            label='Fair Value')

        g1_bid = sns.lineplot(x=time_steps,
                              y=self.bidPrice,
                              ax=ax[0],
                              label='Bid Price',
                              alpha=0.6,
                              linestyle='--')

        g1_ask = sns.lineplot(x=time_steps,
                              y=self.askPrice,
                              ax=ax[0],
                              label='Ask Price',
                              alpha=0.6,
                              linestyle='--')

        g1_2 = sns.lineplot(x=time_steps,
                            y=self.orders,
                            ax=ax[1],
                            label='Inventory',
                            drawstyle='steps-pre')

        g1_3 = sns.lineplot(x=time_steps,
                            y=self.cashflow,
                            ax=ax[2],
                            label='Cash flow',
                            drawstyle='steps-pre')

        g1_4 = sns.lineplot(x=time_steps,
                            y=self.pnl,
                            ax=ax[3],
                            label='PnL',
                            drawstyle='steps-pre')

        plt.show()
