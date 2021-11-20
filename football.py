import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings

filterwarnings('ignore', '.*')

def filter_data(shooting, miscellaneous, home, away):
    home_shooting = shooting.query(f'Squad == "{home}"')
    away_shooting = shooting.query(f'Squad == "{away}"')
    home_misc = miscellaneous.query(f'Squad == "{home}"')
    away_misc = miscellaneous.query(f'Squad == "{away}"')
    HS = home_shooting.get('Sh/90').values[0]
    AS = away_shooting.get('Sh/90').values[0]
    HST = home_shooting.get('SoT/90').values[0]
    AST = away_shooting.get('SoT/90').values[0]
    HY = home_misc.get('CrdY').values[0]/10
    AY = away_misc.get('CrdY').values[0]/10
    HR = home_misc.get('CrdR').values[0]/10
    AR = away_misc.get('CrdR').values[0]/10
    HF = home_misc.get('Fls').values[0]/10
    AF = away_misc.get('Fls').values[0]/10
    HGS = home_shooting.get('G/Sh').values[0]
    AGS = away_shooting.get('G/Sh').values[0]
    HGST = home_shooting.get('G/SoT').values[0]
    AGST = away_shooting.get('G/SoT').values[0]

    return [HS, AS, HST, AST, HY, AY, HR, AR, HF, AF, HGS, AGS, HGST, AGST]

leagues = ['Premier League', 'Championship', 'League One', 'League Two', 'La Liga', 'Segunda Division', 'Bundesliga', '2-Bundesliga', 
'Serie A', 'Serie B', 'Ligue 1', 'Ligue 2', 'Eredivisie', 'Portugal', 'Scottish Premiership', 'Belgium', 'Turkey']

data_urls = ['http://www.football-data.co.uk/mmz4281/2122/E0.csv', 'http://www.football-data.co.uk/mmz4281/2122/E1.csv', 
'http://www.football-data.co.uk/mmz4281/2122/E2.csv', 'http://www.football-data.co.uk/mmz4281/2122/E3.csv', 
'http://www.football-data.co.uk/mmz4281/2122/SP1.csv', 'http://www.football-data.co.uk/mmz4281/2122/SP2.csv', 'http://www.football-data.co.uk/mmz4281/2122/D1.csv', 
'http://www.football-data.co.uk/mmz4281/2122/D2.csv', 'http://www.football-data.co.uk/mmz4281/2122/I1.csv', 'http://www.football-data.co.uk/mmz4281/2122/I2.csv', 
'http://www.football-data.co.uk/mmz4281/2122/F1.csv', 'http://www.football-data.co.uk/mmz4281/2122/F2.csv', 'http://www.football-data.co.uk/mmz4281/2122/N1.csv', 
'http://www.football-data.co.uk/mmz4281/2122/P1.csv', 'http://www.football-data.co.uk/mmz4281/2122/SC0.csv', 'https://www.football-data.co.uk/mmz4281/2122/B1.csv',
 'https://www.football-data.co.uk/mmz4281/2122/T1.csv']

league_urls = ['http://fbref.com/en/comps/9/Premier-League-Stats', 'http://fbref.com/en/comps/10/Championship-Stats', 
'http://fbref.com/en/comps/15/League-One-Stats', 'http://fbref.com/en/comps/16/League-Two-Stats', 'http://fbref.com/en/comps/12/La-Liga-Stats', 
'http://fbref.com/en/comps/17/Segunda-Division-Stats', 'http://fbref.com/en/comps/20/Bundesliga-Stats', 
'http://fbref.com/en/comps/33/2-Bundesliga-Stats', 'http://fbref.com/en/comps/11/Serie-A-Stats', 'http://fbref.com/en/comps/18/Serie-B-Stats', 
'http://fbref.com/en/comps/13/Ligue-1-Stats', 'http://fbref.com/en/comps/60/Ligue-2-Stats', 'http://fbref.com/en/comps/23/Eredivisie-Stats', 
'http://fbref.com/en/comps/32/Primeira-Liga-Stats', 'http://fbref.com/en/comps/40/Scottish-Premiership-Stats', 'https://fbref.com/en/comps/37/Belgian-First-Division-A-Stats',
 'https://fbref.com/en/comps/26/Super-Lig-Stats']

print(f'{leagues}\n')

league = input('League: ')
shooting_df = pd.read_html(league_urls[leagues.index(league)], attrs={'id': 'stats_squads_shooting_for'}, flavor='lxml', header=1)[0]
print(np.asarray(shooting_df.get('Squad')))
x = 0
home_teams = []
away_teams = []
while x < 1:
    home_team = input('Home Team: ')
    if home_team == 'end':
        x += 1
        break
    away_team = input('Away Team: ')
    home_teams.append(home_team)
    away_teams.append(away_team)

criterion = input('Criterion: ')
print('\n')

misc_df = pd.read_html(league_urls[leagues.index(league)], attrs={'id': 'stats_squads_misc_for'}, flavor='lxml', header=1)[0]

df = pd.read_csv(data_urls[leagues.index(league)])

data = pd.DataFrame(df[['FTR', 'HS', 'AS', 'HST', 'AST', 'HY', 'AY', 'HR', 'AR', 'HF', 'AF']])
data['HGS'] = round(df['FTHG'] / df['HS'], 2)
data['AGS'] = round(df['FTAG'] / df['AS'], 2)
data['HGST'] = round(df['FTHG'] / df['HST'], 2)
data['AGST'] = round(df['FTAG'] / df['AST'], 2)

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

if criterion == '' and league in ['Championship', 'Portugal', 'Belgium', '2-Bundesliga', 'Ligue 2', 'Serie A', 'Segunda Division']:
    criterion = 'gini'
elif criterion == '' and league not in ['Championship', 'Portugal', 'Belgium', '2-Bundesliga', 'Ligue 2', 'Serie A', 'Segunda Division']:
    criterion = 'entropy'
else:
    criterion = criterion

X_train, X_test, Y_train, Y_test = train_test_split(data.drop('FTR', axis=1), data['FTR'], random_state=0)

# calculate best max_depth and maybe best of other factors
max_depth_range = list(range(1, 10))
accuracy = []

for depth in max_depth_range:
    clf = RandomForestClassifier(max_depth=depth, random_state=0, criterion=criterion)
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    accuracy.append(score)

max_accuracy = max(accuracy)
depth_to_use = accuracy.index(max_accuracy) + 1
print(f'max_depth is set to {depth_to_use}\n')

min_samples_leaf = list(range(1, 10))
min_samples_split = list(range(2, 10))
accuracy = []

for leaf in min_samples_leaf:
    clf = RandomForestClassifier(max_depth=depth_to_use, random_state=0, criterion=criterion)
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    accuracy.append(score)

max_accuracy = max(accuracy)
min_leaf = accuracy.index(max_accuracy) + 1
print(f'min_samples_leaf is set to {min_leaf}\n')

accuracy = []
for split in min_samples_split:
    clf = RandomForestClassifier(max_depth=depth_to_use, random_state=0, criterion=criterion, min_samples_leaf=min_leaf, min_samples_split=split)
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    accuracy.append(score)

max_accuracy = max(accuracy)
min_split = accuracy.index(max_accuracy) + 2
print(f'min_samples_split is set to {min_split}\n')

clf = RandomForestClassifier(random_state=0, criterion=criterion, max_depth=depth_to_use, min_samples_leaf=min_leaf, min_samples_split=min_split)
clf.fit(X_train, Y_train)

for i in range(len(home_teams)):
    hteam = home_teams[i]
    ateam = away_teams[i]
    prediction_data = filter_data(shooting_df, misc_df, hteam, ateam)
    reshaped_data = np.asarray(prediction_data).reshape(1, -1)
    result = clf.predict(reshaped_data)[0]
    print(f'{hteam} vs {ateam}: {result}')
    classes = [i for i in clf.classes_]
    classes.reverse()
    print(classes)
    probs = clf.predict_proba(reshaped_data)[0]
    probs = [i for i in probs]
    probs.reverse()
    odds = [(1/i) for i in probs]
    print(probs)
    print(f'Odds: {odds}\n')



# result = clf.predict(np.asarray(prediction_data).reshape(1, -1))[0]

# if result == 'H':
#     print('Home Win predicted\n')
# elif result == 'A':
#     print('Away Win predicted\n')
# else:
#     print('Draw predicted\n')

# print(f'{clf.classes_}\n')
# print(f"{clf.predict_proba(np.asarray(prediction_data).reshape(1, -1))[0]}\n")

# importances = pd.DataFrame({'feature':X_train.columns, 'importance':np.round(clf.feature_importances_,2)})
# importances = importances.sort_values('importance', ascending=False)

# print(importances)

print(f'The training data accuracy is: {round(clf.score(X_train, Y_train) * 100, 2)}%\n')
print(f'The test data accuracy is: {round(clf.score(X_test, Y_test) * 100, 2)}%\n')