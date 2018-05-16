#-*- coding: utf-8 -*-

import numpy as np

class ReadCSV():

    def set_csv_file_path(self, csv_file):
        self.csv_file = csv_file
        return (csv_file)

    # Read CSV file and create two vectors: points x and points y
    def read_csv_data(self, csv_file):
        vetor_points_x = []
        vetor_points_y = []
        with open(csv_file, 'r') as csvfile:
            for line in csvfile:
                x, y = line.strip().split(',')
                vetor_points_x.append(float(x))
                vetor_points_y.append(float(y))
        return (vetor_points_x, vetor_points_y)


# class Classify()
#     ...

def main():
    csv_file_path = 'ai-task-132.csv'
    readcsv = ReadCSV()
    csv_file = readcsv.set_csv_file_path(csv_file_path)
    vetor_points_x, vetor_points_y = readcsv.read_csv_data(csv_file=csv_file)
    # print(vetor_points_x, vetor_points_y)

if __name__ == "__main__":
    main()