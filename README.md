Instructions on how to use the sentiment analysis application:

1) Ensure python>=3.6 is used.
2) Run ```pip install -r requirements.txt``` in the working directory and run ```pip install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html``` or check pytorch.org for your version of a stable update. Installation will take a few minutes.
3) Run ```python ./deep_sentiment_analysis.py```
4) Follow the instructions in the application.
5) Outputs are probabilities of negative, neutral or positive class for the given review and the max probability is its predicted class.

Note: model weights may take up to a few minutes to load into your device, ensure some RAM is free to run the application.