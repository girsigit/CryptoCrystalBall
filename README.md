# CryptoCrystalBall - Predict the future of crypto and stock prices

**A framework for predicting entry and exit signals for trading cryptocurrencies (or ETFs, stocks, ...).**

It is more a kind of a 'research' project to find a method of automatically generating trade entry and exit signals than a ready-to-use application. Of course, if I find a suitable and general enough algorithm, it can then be used as a live trading application.

:warning::warning: WARNING: If you use any trade signals from this application for trading with real money, be aware they could be misleading! You could loose your money, be careful and think of what you are doing! I cannot and will not take any responsibility for the generated signals. :warning::warning:

As I have to clean up and document all my code, please be patient that not everything is already here, hopefully I will have it completed by christmas! :christmas_tree:

---

## Brief overview

This is the part for the people who do not want to read much text! So whats the purpose of this repo? The final goal is to generate trade signals like this:

![Chart image](Documentation/Images/Trade_signals_plotted_on_open_price_values.svg)

So whats the way to this goal?

1. Calculate financial indicators on a given training OHLCV dataset. --> [IndicatorCalculator](IndicatorCalculator/README.md)
2. Generate pairs of past ticks data `X-Blocks` and future trade signals `y-data` out of the OHLCV + Indicator training data. --> [XBlockGenerator + YDataGenerator](DataStreamCreator/README.md)
3. Merge them into a shuffled stream of data --> [DataStreamCreator, Todo: write Doku](DataStreamCreator/README.md)
4. Feed the stream to a neural network, which shall learn to predict the signals out of an `X-Block`. --> [Todo: Add the training notebook and the model zoo](nothing.md)
5. Evaluate the model to check if the signals are right.  --> [Todo: Add the testing notebook](nothing.md)
6. Run the model in the live application and have fun :grin:. --> [SignalGenerator, Todo: Has to be adapted](SignalGenerator)

All of these steps are more or less already programmed, I just have to clean the code up so that someone except me understands it, so please be patient!

---
## Training and Testing Data Source

The crypto history data is provided in .csv files on google drive. You can copy them to your own drive using this Link: [Get the data](https://drive.google.com/drive/folders/1HUq1YTD_5N4j6a42ZdchUytdTDYVyXce?usp=sharing)

The important folders are:

- train - Data for training
- test - Data for testing
- real03 - Data for double-testing, for example if the model is trained on the *train* data, the trade algorithm has been developed using the *test* data, this one can be used to evaluate both of them

Most work is done in Jupyter Notebooks in [Google Colab](https://colab.research.google.com/), which is able to read and write data from the Google Drive, but also from the Google Bucket storage.

---
## X_Data_Clusterer

A Jupyter notebook for training a algorithm to cluster X data history frames (generated by the *DataStreamCreator*, see below).

Using this clustering, a historical time-slice can be treated like a tokenized sentence in NLP (if you don't know what this is, see [Link](https://www.geeksforgeeks.org/nlp-how-tokenizing-text-sentence-words-works/)) and therefore powerful ML networks can be applied to it.
