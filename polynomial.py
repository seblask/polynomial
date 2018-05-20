#-*- coding: utf-8 -*-

import numpy as np
import os
import pickle
import argparse

# class CheckFileExist():

class ArgumentParser():
    def __init__(self):
        self.parser()

    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--train', action='store_true', default=False, dest='mode_train',
                            help='train neural network')
        parser.add_argument('-e', '--estimate', action='store_true', default=False, dest='mode_estimate',
                            help='calculate polynomial function f(x)')
        parser.add_argument('-sm', '--save_model', action='store', default='model', dest='save_model',
                            help='set path to save model')
        parser.add_argument('-om', '--open_model', action='store', default='model', dest='open_model',
                            help='set path to open model')
        parser.add_argument('-pd', '--polynomial_degree', action='store', default=2, dest='polynomial_degree',
                            type=int, help='set polynomial degree')
        parser.add_argument('-lr', action='store', default=0.0000001, dest='learning_rate', type=float,
                            help='set learning rate parameter')
        parser.add_argument('-i', action='store', default=1000000, dest='iterations', type=int,
                            help='set number of iterations')
        parser.add_argument('-v', '--version', action='version', version='%(prog)s 1.0')
        results = parser.parse_args()
        return(results)

class ReadCSV():

    def __init__(self, csv_file):
        self.csv_file = csv_file
    # def set_csv_file_path(self, csv_file):
    #     self.csv_file = csv_file
    #     return(csv_file)

    def read_csv_data(self, csv_file):
        '''
        Read CSV file and create two vectors: points x and points y
        '''

        vetor_points_x = []
        vetor_points_y = []
        csv_file = os.path.abspath(csv_file)
        with open(csv_file, 'r') as csvfile:
            for line in csvfile:
                x, y = line.strip().split(',')
                vetor_points_x.append(float(x))
                vetor_points_y.append(float(y))
        return(vetor_points_x, vetor_points_y)

class ModelFileOperations():

    def save_model(self, model, model_path):
        model_path = os.path.abspath(model_path)
        with open(model_path, 'wb') as write_model:
            pickle.dump(model, write_model)
        return(model_path)

    def open_model(self, model_path):
        model_path = os.path.abspath(model_path)
        with open(model_path, 'rb') as read_model:
            model = pickle.load(read_model)
        return(model)

class TrainNN(ReadCSV):

    def __init__(self, csv_file, polynomial_degree):
        '''
        initial parameters
        
        :param csv_file: csv file path
        :param polynomial_degree: int

        set objects:
        vector_points_x, vector_points_y from csv file
        polynomial_degree
        k = length of vector_points_x or vector_points_y
        X = [[1, x1^1, x1^2, ..., x1^n], [1, x2^1, x2^2, ..., x2^n], ..., [1, xk^1, xk^2, ..., xk^n]]
        B - start polynomial coefficient;
        Y = [y1, y2, ..., yk]
        alpha
        '''

        super().__init__(csv_file)
        self.vector_points_x, self.vector_points_y = self.read_csv_data(csv_file)
        self.polynomial_degree = polynomial_degree
        k = len(self.vector_points_x)
        X = []
        x0 = np.ones(k)
        for n in reversed(range(polynomial_degree)):
            x_n = np.array([x**(n+1) for x in self.vector_points_x])
            X.append(x_n)
        X.append(x0)
        self.X = np.array(X).T
        self.B = np.zeros(polynomial_degree+1)
        self.Y = np.array(self.vector_points_y)
        self.alpha = 0.0000000000000000001

    def train(self):
        initial_cost = self.cost_function(self.X, self.Y, self.B)
        B, cost, rmse, r2_score = self.gradient_descent(self.X, self.Y, self.B, self.alpha)
        return(B, cost, rmse, r2_score)

    def cost_function(self, X, Y, B):
        '''
        Cost function
        k = length of vector_points_x or vector_points_y
        :param X: [[1, x1^1, x1^2, ..., x1^n], [1, x2^1, x2^2, ..., x2^n], ..., [1, xk^1, xk^2, ..., xk^n]]
        :param Y: [y1, y2, ..., yk]
        :param B (start polynomial coefficient): [1(1), 1(2), ..., 1(polynomial_degree+1)]
        :return: float
        '''

        m = len(Y)
        J = np.sum((X.dot(B) - Y) ** 2) / (2 * m)
        return(J)

    def rmse(self, Y, Y_pred):
        '''
        Model Evaluation - Root Mean Squared Error
        :param Y: input Y values
        :param Y_pred: predict Y values
        :return: float - the model is more accurate as the value is small
        '''

        rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
        return(rmse)

    def r2_score(self, Y, Y_pred):
        '''
        Model Evaluation - Coefficient of Determination (R2 Score)
        :param Y: input Y values
        :param Y_pred: predict Y values
        :return: float - the model is more accurate as the value is small
        '''

        mean_y = np.mean(Y)
        ss_tot = sum((Y - mean_y) ** 2)
        ss_res = sum((Y - Y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return(r2)

    def gradient_descent(self, X, Y, B, alpha):
        '''
        Gradient Descent
        :param X: input X values
        :param Y: input Y values
        :param B: start polynomial coefficient
        :param alpha: parameter
        :return: B - output polynomial coefficient
                cost, rmse, r2_score - this parameters say about accuracy of model
        '''

        # Y_pred = Y * 2
        # cost = 0
        # rmse = self.rmse(Y, Y_pred)
        # r2_score = self.r2_score(Y, Y_pred)
        m = len(Y)

        for iteration in range(100000):
        # while rmse > 0.001:
            # Hypothesis Values
            h = X.dot(B)
            # difference b/w hypothesis and actual Y
            loss = h - Y
            # gradient calculation
            gradient = X.T.dot(loss) / m
            # changing values of B using gradient
            B = B - alpha * gradient
            # new cost value
            cost = self.cost_function(X, Y, B)
            Y_pred = X.dot(B)
            rmse = self.rmse(Y, Y_pred)
            r2_score = self.r2_score(Y, Y_pred)
            print('iter: '+str(iteration) + '; cost: ' + str(cost))
        return(B, cost, rmse, r2_score)

class Classify():

    def __init__(self, model, x):
        self.model = model
        self.x = x

    def estimate(self):
        polynomial_coefficients = self.model
        polynomial_degree = len(self.model)
        powers = [n for n in reversed(range(polynomial_degree))]
        y = 0

        for power, coefficient in zip(powers, polynomial_coefficients):
            y = y + (coefficient * (self.x**power))
        return(y)



def main():
    results = ArgumentParser().parser()
    if results.mode_train:
        print(results.save_model, results.learning_rate, results.iterations)
    if results.mode_estimate:
        print(results.open_model, results.polynomial_degree)

    # parser = ArgumentParser()
    # results = parser.parser()
    # mode_rec = results.mode_rec
    # mode_srec = results.mode_srec
    # mode_oov = results.mode_oov
    # dictionary_text_file = results.file_dic
    # levenstein_text_file = results.file_lev
    # save_path = results.save_path

    # csv_file_path = 'ai-task-samples.csv'
    # B, cost, rmse, r2_score = TrainNN(csv_file_path, 2).train()
    # ModelFileOperations().save_model(model=B, model_path='model')
    # print(rmse, r2_score)
    # print(B)
    #
    # model = ModelFileOperations().open_model(model_path='model')
    # y = Classify(model=model, x=3).estimate()
    # print(y)

if __name__ == "__main__":
    main()