#!/bin/bash

# Install Python dependencies
pip install -q -U transformers
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U sentencepiece
pip install -q -U youtube-transcript-api==0.6.2
pip install -q -U accelerate
pip install flask 
pip install nltk==3.5.0
pip install xlrd
pip install PyPDF2
pip install keybert
pip install rarfile
pip install openpyxl
pip install python-pptx
pip install python-docx
pip install sense2vec==2.0.1
pip install git+https://github.com/boudinfl/pke.git
python -m spacy download en_core_web_sm

# Download NLTK resources
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
python -m nltk.downloader omw-1.4
python -m nltk.downloader wordnet

# Download Sense2Vec model
wget -O s2v_reddit_2015_md.tar.gz https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
tar -xvf s2v_reddit_2015_md.tar.gz