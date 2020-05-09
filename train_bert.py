from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
import torch
import os

OUTPUT_DIR = "model_output/"
if not os.path.exists(OUTPUT_DIR):
	os.mkdir(OUTPUT_DIR)

logger = logging.getLogger()
labels = ["anger", "anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust","neutral"]
databunch = BertDataBunch(".", ".",
													tokenizer='bert-base-uncased',
													train_file='nlp_train.csv',
													label_file='labels.csv',
													val_file="nlp_valid.csv",
													text_col='text',
													label_col=labels,
													batch_size_per_gpu=16,
													max_seq_length=512,
													multi_gpu=False,
													multi_label=True,
													model_type='bert')

device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=False,
						is_fp16=False,
						multi_label=True,
						logging_steps=50)

learner.fit(epochs=250,
			lr=6e-5,
			schedule_type="warmup_cosine",
			optimizer_type="lamb")

learner.save_model()