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
	thresholds = [0.012, 0.015, 0.02, 0.25, 0.03,0.035,0.039, 0.04, 0.045, 0.05, 0.06, 0.07, 0.1]
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
	# print(len(inputs))
	# print("dta")
	# print(len(data))
	for th in thresholds:
		correct = 0
		# print(list(inputs.values())[0])
		outputs = []
		for out in multiple_predictions:
			temp = []
			for emotion in out:
				if emotion[1] > th:  # greater than threshold
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
	parser.add_argument("--model_dir",default="D:\\UTD\\Assignment\\NLP\\project\\model_output\\5epochs_15", help="path to output dir")
	parser.add_argument("--test_csv", default="D:\\UTD\\Assignment\\NLP\\project\\nlp_train.csv")
	args = parser.parse_args()
	threshold(args.model_dir, args.test_csv)