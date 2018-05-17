#-*- coding: utf-8 -*-

import numpy as np

class ReadCSV():

    def set_csv_file_path(self, csv_file):
        self.csv_file = csv_file
        return (csv_file)

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
        return (vetor_points_x, vetor_points_y)

class TrainNN(ReadCSV):

    def __init__(self, csv_file, polynomial_degree):
        '''
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
        self.B = np.zeros(k)
        self.Y = np.array(self.vector_points_y)
        self.alpha = 0.0001



# class Classify()
#     ...

def main():
    csv_file_path = 'ai-task-132.csv'
    readcsv = ReadCSV()
    csv_file = readcsv.set_csv_file_path(csv_file_path)
    # vetor_points_x, vetor_points_y = readcsv.read_csv_data(csv_file=csv_file)
    TrainNN(csv_file, 3)



if __name__ == "__main__":
    main()