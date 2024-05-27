sync:
	aws s3 sync s3://soundx-audio-dataset data/raw --profile soundx --delete

train:
	python src/multi_train.py

eval:
	python src/multi_eval.py

streamlit:
	streamlit run Soundx.py