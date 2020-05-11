## Multi Emotion Detection from Text using BERT ##

#### Data Preparation ####
The model expects data to be in the csv file. So, you first need to convert json files into csv files. If you have data in any other format, you may need to modify the code.

To generate the csv files run the following command.

```
python .\data_generator.py --file=D:\UTD\Assignment\NLP\project\nlp_test.json --csvfile=D:\UTD\Assignment\NLP\project\nlp_test.csv
```

Here ```--csvfile``` represents where to store the converted file.


#### Training ####
Once you have the files in the required format, you can start training. You may want to change the parameters. I tried with multiple parameters and the file contains ones that gave the best result.
```
python train_bert.py --epochs=20
```

You can find all available options by running following command.

```
python train_bert.py --help
```


#### Inference ####
Once you have the trained model, you can run the inference on test csv files. Note that as of now, this script requires annotated data to compute the metrics. But it can easily be modified to generate output only.

```
python inference.py --test_csv=D:\\UTD\\Assignment\\NLP\\project\\nlp_valid.csv --model_dir=D:\\UTD\\Assignment\\NLP\\project\\model_output\\3_finetune_e20
```

If ```--evaluation``` is set to true, it will output various metrics.

**Threshold**
One important factor here is to find the optimal threshold for the confidence score. I tested various threshold and found 0.0017 to give the best results for the above specified model. If you train your own model, you may want to run ```find_threshold.py``` to find the best threshold.

#### Model Evaluation ####
