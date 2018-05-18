#-*- coding: utf-8 -*-

import numpy as np
import os
import pickle

# class CheckFileExist():

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
        self.alpha = 0.0001

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

        Y_pred = Y * 2
        cost = 0
        rmse = self.rmse(Y, Y_pred)
        r2_score = self.r2_score(Y, Y_pred)
        m = len(Y)

        while rmse > 0.001:
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
    # csv_file_path = 'ai-task-132.csv'
    # B, cost, rmse, r2_score = TrainNN(csv_file_path, 2).train()
    # ModelFileOperations().save_model(model=B, model_path='model')

    model = ModelFileOperations().open_model(model_path='model')
    y = Classify(model=model, x=3).estimate()
    print(y)

if __name__ == "__main__":
    main()