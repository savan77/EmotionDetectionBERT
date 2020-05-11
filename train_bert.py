# Training script for bert

from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
import torch
import os
import argparse

OUTPUT_DIR = "model_output/"

def train(args):
	if args.is_onepanel:
		args.out_dir = os.path.join("/onepanel/output/",args.out_dir)
	if not os.path.exists(args.out_dir):
		os.mkdir(args.out_dir)

	logger = logging.getLogger()
	labels = ["anger", "anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"]
	databunch = BertDataBunch(".", ".",
								tokenizer=args.pretrained_model,
								train_file='nlp_train.csv',
								label_file='labels.csv',
								val_file="nlp_valid.csv",
								text_col='text',
								label_col=labels,
								batch_size_per_gpu=args.batch_size,
								max_seq_length=512,
								multi_gpu=False,
								multi_label=True,
								model_type='bert')

	device_cuda = torch.device("cuda")
	metrics = [{'name': 'accuracy', 'function': accuracy}]

	learner = BertLearner.from_pretrained_model(
							databunch,
							pretrained_path=args.pretrained_model,
							metrics=metrics,
							device=device_cuda,
							logger=logger,
							output_dir=args.out_dir,
							finetuned_wgts_path=None,
							warmup_steps=200,
							multi_gpu=False,
							is_fp16=False,
							multi_label=True,
							logging_steps=10)

	learner.fit(epochs=args.epochs,
				lr=2e-3,
				schedule_type="warmup_cosine_hard_restarts",
				optimizer_type="lamb")
				# validate=True)
	learner.save_model()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--pretrained_model", default="bert-base-uncased", help="path to a pretrained model")
	parser.add_argument("--out_dir",default="model_output/", help="path to output dir")
	parser.add_argument("--is_onepanel", default=False, type=bool, help="train on onepanel cloud")
	parser.add_argument("--epochs", default=15, type=int)
	parser.add_argument("--batch_size", default=10, type=int)
	args = parser.parse_args()
	train(args)
