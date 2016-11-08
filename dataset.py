import csv
import numpy as np

# TODO: Sane helpstrings
class Dataset():

    def __init__(self, filename):
        self.csv_filename = filename
        self.data = None
        self.features = None
        self.open_csv(self.csv_filename)


    def open_csv(self, csv_filename):
        with open(csv_filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            # Getting the first row as the names for our NumPy array
            col_names = reader.__next__()
            print(col_names)
            datatypes = self.generate_datatypes(col_names)
            self.data = np.array([], dtype=datatypes)
            self.features = np.empty((0, 17), float)
            print(self.features)
            for row in reader:
                row = [x.replace(",",".") for x in row]
                row_printed = self.print_row(row)
                if row_printed.strip() == "":
                    continue

                arr = np.array(tuple(row), dtype=datatypes)
                self.data = np.append(self.data, arr)

                new_features = []
                for el in row[2:]:
                    new_features.append(float(el))
                self.features = np.vstack([self.features, new_features])



    # Hardcoded for now, sadly
    def generate_datatypes(self, names):
        datatypes = [(names[0], 'i4'),
                     (names[1], 'S10'),
                     (names[2], 'i4'),
                     (names[3], 'i4'),
                     (names[4], 'f4'),
                     (names[5], 'i4'),
                     (names[6], 'i4'),
                     (names[7], 'f4'),
                     (names[8], 'i4'),
                     ]
        for x in range(9, len(names)):
            datatypes.append((names[x], 'f4'))
        return datatypes

    def shuffle(self):
        np.random.shuffle(self.data)

    def print_row(self, row):
        return " ".join(row)

if __name__ == "__main__":
    eNode = Dataset("KRD.csv")
    print(eNode.data)


