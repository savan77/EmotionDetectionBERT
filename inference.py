from fast_bert.prediction import BertClassificationPredictor
import argparse


def run(model,csvs, threshold):
	labels = ["anger", "anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"]

	predictor = BertClassificationPredictor(
					model_path=args.model_dir,
					label_path="D:\\UTD\\Assignment\\NLP\\project\\", # location for labels.csv file
					multi_label=False,
					model_type='bert',
					do_lower_case=False)
	
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
	outputs = []
	for out in multiple_predictions:
		temp = []
		for emotion in out:
			if emotion[1] > threshold:  # greater than threshold
				temp.append(emotion[0])
		outputs.append(temp)
	print("****************\n")
	print("**   predictions:  \n")
	print(outputs)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_dir",default="D:\\UTD\\Assignment\\NLP\\project\\model_output\\5", help="path to output dir")
	parser.add_argument("--test_csv", default="D:\\UTD\\Assignment\\NLP\\project\\nlp_train.csv")
	parser.add_argument("--threshold", default=0.04, type=float)
	args = parser.parse_args()
	threshold(args.model_dir, args.test_csv, args.threshold)