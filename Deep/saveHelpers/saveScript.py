import csv
import os

def save_losses(save_name, header = None, values=['0','0']):
    f = open(save_name, 'a')
    writer = csv.writer(f)
    if header is not None:
        writer.writerow(header)
    writer.writerow(values)