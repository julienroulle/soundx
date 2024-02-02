sync:
	aws s3 sync s3://soundx-audio-dataset data/raw --profile soundx --delete

train:
	aws s3 sync s3://soundx-audio-dataset data/raw --profile soundx --delete
	python train.py
