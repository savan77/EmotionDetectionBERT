# Script to generate inference ofr a given csv file

from fast_bert.prediction import BertClassificationPredictor
import argparse
import csv
import pandas as pd
import os
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from pprint import pprint

# run inference on the csv file provided using the trained model
def run(model,csvs, threshold, evaluation):
	labels = ["anger", "anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"]

	predictor = BertClassificationPredictor(
					model_path=args.model_dir,
					label_path="D:\\UTD\\Assignment\\NLP\\project\\", # location for labels.csv file
					multi_label=False,
					model_type='bert',
					do_lower_case=False)
	
	inputs = {} 
	ids = []
	data = pd.read_csv(csvs)
	# print(data.head())
	for idx, row in data.iterrows():
		temp = []
		for label in labels:
			if row[label] == 1:
				temp.append(label)
		inputs[row['text']] = temp
		ids.append(row['id'])

	multiple_predictions = predictor.predict_batch(list(inputs.keys()))
	outputs = []
	out_file = open(os.path.join(os.path.dirname(csvs),"model_output.csv"), "w", encoding="utf-8", newline="")
	csv_writer = csv.writer(out_file)
	csv_writer.writerow(["id","text", "emotions", "target"])

	for i, out in enumerate(multiple_predictions):
		temp = []
		for emotion in out:
			if emotion[1] > threshold:  # greater than threshold
				temp.append(emotion[0])
		csv_writer.writerow([ids[i],list(inputs.keys())[i],temp,list(inputs.values())[i] ])
		outputs.append(temp)

	print("****************\n")
	print("Predictions saved in a file: ", os.path.join(os.path.dirname(csvs),"model_output.csv"))
	if evaluation:
		print("\n\n Running Model Evaluation\n")
		y_true = list(inputs.values())
		y_pred = outputs
		y_true_encoded = MultiLabelBinarizer().fit_transform(y_true)
		y_pred_encoded = MultiLabelBinarizer().fit_transform(y_pred)
		pprint(classification_report(y_true_encoded, y_pred_encoded))
		pprint(classification_report(y_true_encoded, y_pred_encoded, target_names=labels))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_dir",default="D:\\UTD\\Assignment\\NLP\\project\\model_output\\3_finetune_e20", help="path to output dir")
	parser.add_argument("--test_csv", default="D:\\UTD\\Assignment\\NLP\\project\\nlp_test.csv")
	parser.add_argument("--threshold", default=0.0017, type=float)
	parser.add_argument("--writeto_file", default=True)
	parser.add_argument("--evaluation", default=True)
	args = parser.parse_args()
	run(args.model_dir, args.test_csv, args.threshold, args.evaluation)