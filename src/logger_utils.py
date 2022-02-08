import csv
import json
import os

def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for filter in logger.filters[:]:
        logger.removeFilters(filter)

def save_dict(log_path, name, data):
    path = os.path.join(log_path, f'{name}.json')
    f = open(path, 'w')
    json.dump(vars(data), f)
    f.close()

def append_csv(file_path, *row):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    with open(file_path, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=';', quatochar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(row)
