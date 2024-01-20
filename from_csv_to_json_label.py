import csv
import json

def csv_to_json(csv_file, json_file):
    data = {}
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        _ = next(csv_reader)

        for row in csv_reader:
            filename = row[0]
            labels = row[3:]
            data[filename] = labels

    with open(json_file, 'w') as json_output:
        json.dump(data, json_output, indent=4)
        

if __name__ == "__main__":
    csv_input_file = '/data/disk2/vinhnguyen/AnyFace/data/labels_train.csv'
    json_output_file = '/data/disk2/vinhnguyen/AnyFace/data/labels_train.json'

    # csv_input_file = '/data/disk2/vinhnguyen/AnyFace/data/labels_valid.csv'
    # json_output_file = '/data/disk2/vinhnguyen/AnyFace/data/labels_valid.json'
    
    
    csv_to_json(csv_input_file, json_output_file)
