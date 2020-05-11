######
# Script to find best possible threshold for the confidence.
#

from fast_bert.prediction import BertClassificationPredictor
import argparse
import csv
import pandas as pd

def threshold(model, csvs):
	labels = ["anger", "anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"]

	predictor = BertClassificationPredictor(
					model_path=args.model_dir,
					label_path="D:\\UTD\\Assignment\\NLP\\project\\", # location for labels.csv file
					multi_label=False,
					model_type='bert',
					do_lower_case=False)
	thresholds = [0.0005,0.00077,0.00079,0.00083,0.00087,0.0009,0.00093,0.00095,0.00099,0.001,0.0012,0.0015,0.00155,0.0016,0.00166,0.0017,0.0019,0.002,0.0021,0.0023,0.0025,0.0028,0.003,0.0035,0.0032,0.0037,0.004,0.0045,0.0047,0.0041,0.005,0.0053,0.0055,0.0062,0.009, 0.007, 0.01, 0.011,0.013,0.014,0.012, 0.015, 0.02, 0.25, 0.03,0.035,0.039]
	# targets = []
	inputs = {} 
	data = pd.read_csv(csvs)
	# print(data.head())
	for idx, row in data.iterrows():
		temp = []
		for label in labels:
			if row[label] == 1:
				temp.append(label)
		inputs[row['text']] = temp

	multiple_predictions = predictor.predict_batch(list(inputs.keys()))
	threshold_accs = {}
	
	for th in thresholds:
		correct = 0
		# print(list(inputs.values())[0])
		outputs = []
		for out in multiple_predictions:
			temp = []
			for emotion in out:
				if emotion[1] >= th:  # greater than threshold
					temp.append(emotion[0])
			outputs.append(temp)
		# print(outputs[0])
		for i in range(len(inputs)):
			if (set(outputs[i]) == set(list(inputs.values())[i])):
				correct += 1
		print("Threshold: ", th, "Correct: ", correct)
		threshold_accs[str(th)] = correct/len(inputs)
	print(threshold_accs)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_dir",default="D:\\UTD\\Assignment\\NLP\\project\\model_output\\3_finetune_e20", help="path to output dir")
	parser.add_argument("--test_csv", default="D:\\UTD\\Assignment\\NLP\\project\\nlp_test.csv")
	args = parser.parse_args()
	threshold(args.model_dir, args.test_csv)