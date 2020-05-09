from fast_bert.prediction import BertClassificationPredictor
OUTPUT_DIR = "model_output/"
LABEL_PATH = "."
MODEL_PATH = OUTPUT_DIR+'1'

predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path=LABEL_PATH, # location for labels.csv file
				multi_label=False,
				model_type='xlnet',
				do_lower_case=False)


# Batch predictions
texts = [
	"this is the first text",
	"this is the second text"
	]

multiple_predictions = predictor.predict_batch(texts)
print(multiple_predictions)