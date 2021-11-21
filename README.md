# Football prediction AI

## About

This is a project I created that attempts to predict the outcome of football matches in many European leagues, based on statistics and Artificial Intelligence. It works by using a Random Forest classifier to predict whether two teams facing off will result in a home win, a draw or an away win. 

This prediction is based on specific stats of the teams as well as match data for the current season of the selected league.

The original script was just one that was run from the terminal and interacted with through there. However, I turned it into a module for use in other scripts if that was wanted.

> Note: this is not meant to win you a fountain of money through sports betting. You will most likely lose money, because, for however well this AI predicts results off stats, it cannot factor in intangible factors such as team form, player injuries, new-manager bounce, etc. 

## Built with
This project was built with the following languages and packages
- Python 3.8.10+
- Pandas, which is used for organising and filtering data with ease, as well as scraping HTML tables holding the stats
- Scikit-Learn, from which I use the `train_test_split` function and the `RandomForestClassifier` class for the predictions.
- Numpy, which I use to reshape the data collected by Pandas into the correct shape for the `RandomForestClassifier` class.
- Warnings, which I just use to silence some warnings that Scikit-Learn shows.

## How it works
There are two classes that make up the module: the `Data` class and the `Predictor` class.

### Data
The `Data` class is used for gathering the match data and the team data for the league that you wish to predict results for. The main method is the `get_data()` function. This will get you the league data and match data. 

The `Data` class also has a method called `select_criterion()`, which automatically chooses either `gini` or `entropy` as the criterion for the `Predictor` class. Otherwise, it can be specified when the `Predictor` class is instantiated.

### Predictor

The `Predictor` class takes the league data and match data gathered by the `Data` class. It also takes a parameter `criterion`, which can be manually specified or automatically selected.

There is also a method called `hyperparameter_tuning()`. Usually, AI models will use Scikit-Learn's  `GridSearchCV` class to select the best combination of hyperparameters for the `RandomForestClassifier`. However, this slows down the module immensely, and so my `hyperparameter_tuning()` method works less effectively but is far quicker, so much so that I preferred to use it to `GridSearchCV`.

## Disclosure
> Once again, this is not meant to be successful at predicting sports results. It may help somewhat, but be aware that there are external factors that influence results that can't be used in the prediction.