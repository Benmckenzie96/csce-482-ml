import json
import csv

# input data
json_file = open("org_data.json")
json_data = json.load(json_file)
json_file.close()

#data = json.loads(json_data)
data = json_data

tsv_file = open("org_data.tsv", "w")
tsv_writer = csv.writer(tsv_file, delimiter='\t')

tsv_writer.writerow(data[0].keys()) # write the header

for row in data: # write data rows
    tsv_writer.writerow(row.values())

tsv_file.close()