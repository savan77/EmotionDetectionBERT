###
# Generate data (in csv format) to train the BERT model

import csv
import json
import argparse
import os

# store emotions as a one hot encodings
def create_model(llabels):
	llist = [0]* 12
	for em, val in enumerate(llabels.values()):
		llist[em] = 1 if val else 0
	return llist

# generate and write to csv file
def generate_csv(file, csvfile):
	data= open(file,"r")
	out = open(csvfile, "w", encoding="utf-8", newline="")
	writer = csv.writer(out)
	writer.writerow(["id", "text", "anger", "anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"])
	data = json.load(data)
	idd = 0
	for i,v in data.items():
		bin_vector = create_model(v['emotion'])
		writer.writerow([idd,v['body']]+bin_vector)
		idd += 1
	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--file", default="nlp_train.json", type=str)
	parser.add_argument("--csvfile", default="nlp_train.csv", type=str)
	args = parser.parse_args()
	generate_csv(args.file, args.csvfile)