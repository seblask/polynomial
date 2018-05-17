#-*- coding: utf-8 -*-

import numpy as np

class ReadCSV():

    def set_csv_file_path(self, csv_file):
        self.csv_file = csv_file
        return(csv_file)

    def read_csv_data(self, csv_file):
        '''
        Read CSV file and create two vectors: points x and points y
        '''

        vetor_points_x = []
        vetor_points_y = []
        with open(csv_file, 'r') as csvfile:
            for line in csvfile:
                x, y = line.strip().split(',')
                vetor_points_x.append(float(x))
                vetor_points_y.append(float(y))
        return(vetor_points_x, vetor_points_y)

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

        self.vector_points_x, self.vector_points_y = self.read_csv_data(csv_file)
        self.polynomial_degree = polynomial_degree
        k = len(self.vector_points_x)
        X = []
        x0 = np.ones(k)
        X.append(x0)
        for n in range(polynomial_degree):
            x_n = np.array([x**(n+1) for x in self.vector_points_x])
            X.append(x_n)
        self.X = np.array(X).T
        self.B = np.zeros(polynomial_degree+1)
        self.Y = np.array(self.vector_points_y)
        self.alpha = 0.0001

    def train(self):
        initial_cost = self.cost_function(self.X, self.Y, self.B)

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

    # Model Evaluation - R2 Score
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

# class Classify()
#     ...

def main():
    csv_file_path = 'ai-task-132.csv'
    readcsv = ReadCSV()
    csv_file = readcsv.set_csv_file_path(csv_file_path)
    # vetor_points_x, vetor_points_y = readcsv.read_csv_data(csv_file=csv_file)
    TrainNN(csv_file, 2).train()


if __name__ == "__main__":
    main()