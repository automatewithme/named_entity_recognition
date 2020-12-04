glove:
	mkdir data
	wget -P ./data/ "https://github.com/hse-aml/natural-language-processing/releases/download/week2/train.txt"
	wget -P ./data/ "https://github.com/hse-aml/natural-language-processing/releases/download/week2/validation.txt"
	wget -P ./data/ "https://github.com/hse-aml/natural-language-processing/releases/download/week2/test.txt"
	wget "https://raw.githubusercontent.com/hse-aml/natural-language-processing/master/week2 -O evaluation.py"
	wget "https://raw.githubusercontent.com/hse-aml/natural-language-processing/master/common -O requirements_colab.txt"
	pip install -r requirements_colab.txt --force-reinstall

run:
	python build_data.py
	python train.py
	python evaluate.py