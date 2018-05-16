#-*- coding: utf-8 -*-

import numpy as np
import csv

class Train():

    def main(self):
        self.read_csv_data()

    def set_csv_file_path(self, csv_file):
        self.csv_file = csv_file
        return (csv_file)

    def read_csv_data(self):
        vetor_points_x = []
        vetor_points_y = []
        with open(self.csv_file, 'r') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                vetor_points_x.append(float(row[0]))
                vetor_points_y.append(float(row[1]))
        return (vetor_points_x, vetor_points_y)


# class Classify()
#     ...

def main():
    csv_file_path = 'ai-task-132.csv'
    train = Train()
    train.set_csv_file_path(csv_file_path)
    vetor_points_x, vetor_points_y = train.read_csv_data()

if __name__ == "__main__":
    main()