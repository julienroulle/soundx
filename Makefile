sync:
	aws s3 sync s3://soundx-audio-dataset data/raw --profile soundx --delete

train:
	python src/multi_train.py

eval:
	python src/multi_eval.py

streamlit:
	streamlit run Soundx.py

export-requirements:
	poetry export -f requirements.txt --output requirements.txt

modal-deploy:
	modal deploy modal_infra.py