import csv
import numpy as np

filename = "KRD.csv"
with open(filename,newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for row in reader:
        print("\n".join(row))

'''Opens the csv, supposedly separated by russian excel's disgusting ;-delimiter,
Probably should also replace all the comma-separators with dots along the way'''
def open_csv(csv_filename):
    csvfile = open(csv_filename, 'r')
    reader = csv.reader(csvfile, delimiter=';')
    return reader

# Хочу найти способ, чтобы выкидывало просто ридер файла, на закрытии программы он бы закрывался
# Но я чет туплю и не знаю, как его передавать
# Ок, ладно, пофиг, сделаю пока в одном методе
# бтв, точно, надо бы завести гит же, блин
# Забыл уже напрочь, как его юзать х)

if __name__ == "__main__":
    csv_file = open_csv("KRD.csv")
    for row in csv_file:
        print(" ".join(row))


