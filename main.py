# ADM Homework 2 - Mohammadreza Mowlai, Flaminia Spasiano, Andrea Baldino
#### Libraries used


import pandas as pd
#import importlib
#importlib.reload(functions)
import functions
# For RQ1, RQ3, CRQ1 
import matplotlib.pyplot as plt
#CRQ2
from matplotlib.patches import ConnectionPatch
from matplotlib.patches import Arc
# For RQ3
import numpy as np
import seaborn as sns

# For RQ3
from collections import defaultdict
from datetime import datetime
from datetime import date


# ## (1) We import the datasets we need


team = pd.read_json(r"teams.json")
coach=pd.read_json(r"coaches.json")
player= pd.read_json(r"players.json")


event_Spain=pd.read_json(r"events_Spain.json")


event_England=pd.read_json(r"events_England.json")


match_England=pd.read_json(r"matches_England.json")
match_Spain=pd.read_json(r"matches_Spain.json")


match_Italy = pd.read_json(r"matches_Italy.json")


event_Italy = pd.read_json(r"events_Italy.json")


# ## (2) We clean a bit the dataset of the Event in England 


event_England= functions.CleanEngEvents(event_England)


# #### Exctracting the postition columns and the positions and tags, and separate their elements 


# #### In here we create the lists of the staerting and enging posisions (for CRQ2) and the lists of the tags refered to each event (very usefull to filter events given a tag)


#this is what we get
event_England


# ## (3) We create a dataframe (engTeams) of teams who actually played in the Premier Legue, and a list of only the teamId of those teams.


# #### We create these because they are usefull to see all the info concerning a English team all together


engTeams = {} #a dict
eng=[]
for i in range(len(team)):
    if (team["area"][i]["id"] == '0' and team['type'][i] == "club"):
        eng.append(team['wyId'][i])
        engTeams[team['wyId'][i]] = { 'name': team.name[i], 'officialName': team.officialName[i], 'country': team.area[i]['name'] }



# ## (4) We create a dataset of ONLY english players, who actually played in the Premier Legue.


# We filter all the players on only those whose teamId is in the list of the english teams Id
eng_players = player.loc[player['currentTeamId'].isin(eng)]


#We can drop the non-important colums
eng_players = eng_players.drop(columns =["passportArea", "weight", "birthDate", "role", "birthArea", "shortName", "currentNationalTeamId", 'foot'])

eng_players.rename(columns={"wyId": "playerId"}, inplace= True)

# reindexing
pd.DataFrame.reset_index(eng_players, inplace= True)
eng_players.drop(columns=["index"], inplace= True)


# ## We clean Matches England


Match_England= match_England.copy()


match_England= functions.cleaningMatches(match_England)


# # [RQ1]


# ## Who wants to a champion?


# #### We sort the dataset on the gameweek (not necessary)


DF= match_England.copy()


# #### We create a dict whose keys are the TeamId (England and Wales teams (they both have area id= 0) excluding the national teams), and for every key the values are a list of 39 zeros (1 for every week, starting from the week 0). This dict will be updated with the right week-score and will become the dataset of all the scores


engTableNGW = {}
for i in range(len(team["area"])):
        if (team["area"][i]["id"] == "0" and team['type'][i]== "club"):
            engTableNGW[team['wyId'][i]]= [0 for i in range(39)]


scores= functions.CountingPoints(DF, engTableNGW)


scores


# #### Here we rename all the columns with the corresponding Team name (instead of the Team Id)


for col in list(scores.drop(columns={'gameweek'}).columns):
    scores.rename( columns = {col: engTeams[col]['name']}, inplace = True)


# #### The plot


plt.style.use('seaborn-darkgrid') #style used

#size of the figure
plt.figure(figsize=(15,10)) 

# multiple line plot
num=0 
for column in scores.drop('gameweek', axis=1):
    num+=1
    plt.plot(scores['gameweek'], scores[column], linewidth = 2, marker = "o", markersize = 2, alpha = 0.9, label = column)

# the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0,frameon = False, prop={'size': 15})

# Add titles
plt.title("Premier League Table 2017/2018", loc = 'center', fontsize = 30, fontweight = 3, color = 'darkviolet')
plt.xlabel("Weeks", fontsize = 30)
plt.ylabel("Points", fontsize = 30)


# # [RQ2]


# ### Is there a home-field advantage? 


teams=functions.cleaningTeams(team)
matches_England= match_England.copy()
events_England= event_England.copy()
players=functions.cleaningPlayers(player)
coaches= coach.copy()


events_Spain = functions.CleanEngEvents(event_Spain)


matches_Spain= functions.cleaningMatches(match_Spain)


# ## TABLE


tableEng = functions.creatingTable(teams, matches_England)


tableEsp = functions.creatingTable(teams, matches_Spain)


tableEng


# ## Contingency Tables


# We are going to create 5 contingency tables for the weakest teams of the championship, picking the five teams from the bottom. We are going to aggregate the contingency tables calculated. We have removed the matches that those teams played among themselves: in this way we remove every grade on dependency within the sampled date. The obtained table is the final contengecy table which will be used for performing the Chi-squared test ('overall test'). 


dictTeams1 = {'Huddersfield Town': 1673,'Southampton': 1619,'Stoke City': 1639,'Swansea City': 10531,'West Bromwich Albion': 1627 }


dictTeams2 = {'Manchester City': 1625,'Watford': 1644,'Southampton': 1619,'Chelsea': 1610,'Arsenal': 1609 }


pd.DataFrame(pd.Series(dictTeams1))


teamsList = []
for key,value in dictTeams2.items():
        teamsList.append(value) 


teamsList


functions.singleContingencyTable(1673, tableEng)


teams[teams.teamId.isin(teamsList) ][['name', 'teamId']]


def subTable(tmLst):
    # This function creates a table within a selected numbers of teams, just considering the matches between thierselves.
    # INPUT: Array of teamIds
    # OUTPUT: Teams Ranking

    # Filter on the matches between the teams
    strCond = ""

    for el in tmLst:
        strCond = strCond +"| (matches_England.homeTeamId == " + str(el) + ") "
    strCond = strCond[2:-1]

    exec( 'global subTableMatches; subTableMatches = matches_England[ ('+ strCond +') & (matches_England.awayTeamId.isin(tmLst))]' )

    # Creating SUB TABLE between the teams
    # Retrieving Home/Away Statistics & Points
    homeMatches = pd.merge(teams.filter(['name', 'teamId', 'championshipId']), subTableMatches, left_on='teamId', right_on='homeTeamId', how='inner')
    awayMatches = pd.merge(teams.filter(['name', 'teamId', 'championshipId']), subTableMatches, left_on='teamId', right_on='awayTeamId', how='inner')
    hm = homeMatches[['teamId', 'name','homeWin','homeDraw','homeLoss','homePoints' ]].groupby(['teamId','name']).sum()
    am = awayMatches[['teamId', 'name','awayWin','awayDraw','awayLoss','awayPoints' ]].groupby(['teamId','name']).sum()

    # The Final Sub Table
    subTable = pd.merge(hm, am, on='teamId' )

    # Calculating Total Points
    subTable['totalPoints'] = subTable['homePoints'] + subTable['awayPoints']
    subTable['totalWin']    = subTable['homeWin']    + subTable['awayWin']
    subTable['totalDraw']   = subTable['homeDraw']   + subTable['awayDraw']
    subTable['totalLoss']   = subTable['homeLoss']   + subTable['awayLoss']

    # Joining teams' names
    subTable = pd.merge(teams, subTable, left_on='teamId', right_on='teamId', how='inner')[['teamId','name','totalWin','totalDraw','totalLoss','homeWin','homeDraw','homeLoss','awayWin','awayDraw','awayLoss','homePoints','awayPoints','totalPoints']]

    # Sorting the table
    subTable.sort_values("totalPoints", ascending=False, inplace=True)

    #Reindexing
    subTable.reset_index(inplace=True)

    #Dropping columns
    subTable.drop(columns=['index'], inplace=True)

    return subTable



def contingencyTable(tmLst, table):
    # This function creates the contingency table for Chii-Squared test within a selected numbers of teams.
    # It doesn't tanke in consideration the matches between the teams involved in the test.
    # INPUT: Array of teamIds, Teams Ranking of the set of teams
    # OUTPUT: Chi-Squared Contingency Table
    totTable = [ [0,0,0],[0,0,0] ]
    intTable = [ [0,0,0],[0,0,0] ] #Intersection Table: matches between the teams in tmLst
    ctTable  = [ [0,0,0],[0,0,0] ] #Contingency Table: we will use this for the test

    # Global Statistics
    for tm in tmLst:
        tb = functions.singleContingencyTable(tm, table)
        tb.reset_index(inplace=True)
        totTable[0][0] = totTable[0][0] + tb[tb['typeVenue'] == 'home']['win'].loc[0]
        totTable[0][1] = totTable[0][1] + tb[tb['typeVenue'] == 'home']['draw'].loc[0]
        totTable[0][2] = totTable[0][2] + tb[tb['typeVenue'] == 'home']['loss'].loc[0]
        totTable[1][0] = totTable[1][0] + tb[tb['typeVenue'] == 'away']['win'].loc[1]
        totTable[1][1] = totTable[1][1] + tb[tb['typeVenue'] == 'away']['draw'].loc[1]
        totTable[1][2] = totTable[1][2] + tb[tb['typeVenue'] == 'away']['loss'].loc[1]

    # Intra Statistics: matches between the teams in tmLst
    subsetTable = subTable(tmLst) # The table intra the teams selected
    for tm in tmLst:
        tb = functions.singleContingencyTable(tm, subsetTable)
        tb.reset_index(inplace=True)
        intTable[0][0] = intTable[0][0] + tb[tb['typeVenue'] == 'home']['win'].loc[0]
        intTable[0][1] = intTable[0][1] + tb[tb['typeVenue'] == 'home']['draw'].loc[0]
        intTable[0][2] = intTable[0][2] + tb[tb['typeVenue'] == 'home']['loss'].loc[0]
        intTable[1][0] = intTable[1][0] + tb[tb['typeVenue'] == 'away']['win'].loc[1]
        intTable[1][1] = intTable[1][1] + tb[tb['typeVenue'] == 'away']['draw'].loc[1]
        intTable[1][2] = intTable[1][2] + tb[tb['typeVenue'] == 'away']['loss'].loc[1]

    # Matches of selected teams against other teams of the championship
    for i in range(0,2):
        for j in range(0,3):
            ctTable[i][j] = totTable[i][j] - intTable[i][j]

    return ctTable



subTable(teamsList)


functions.singleContingencyTable(1609, subTable(teamsList))


ct = contingencyTable(teamsList, tableEng)


ct


# ## Pearson’s Chi-Squared Test
# The Chi-Squared test is a **statistical hypothesis test** that assumes (the null hypothesis) that the observed frequencies for a categorical variable match the expected frequencies for the categorical variable. The test calculates a statistic that has a chi-squared distribution, named for the Greek capital letter Chi (X).
# 
# Given a random variable, shareable in groups, the number of observations for a category (such as male and female) may or may not the same. We can calculate the expected frequency of observations in each Interest group and see whether the partitioning of interests by the category results in similar or different frequencies.
# 
# The Chi-Squared test does this for a **contingency table**, first calculating the expected frequencies for the groups, then determining whether the division of the groups, called the observed frequencies, matches the expected frequencies.
# 
# The result of the test is a **test statistic** that has a chi-squared distribution and can be interpreted to reject or fail to reject the assumption or null hypothesis that the observed and expected frequencies are the same.
# 
# When observed frequency is far from the expected frequency, the corresponding term in the sum is large; when the two are close, this term is small. Large values of X^2 indicate that observed and expected frequencies are far apart. Small values of X^2 mean the opposite: observeds are close to expecteds. So X^2 does give a measure of the distance between observed and expected frequencies.
# 
# The variables are considered independent if the observed and expected frequencies are similar, that the levels of the variables do not interact, are not dependent.
# 
# The chi-square test of independence works by comparing the categorically coded data that you have collected (known as the observed frequencies) with the frequencies that you would expect to get in each cell of a table by chance alone (known as the expected frequencies).
# 
# We can interpret the test statistic in the context of the chi-squared distribution with the requisite number of degress of freedom as follows:
# 
#  - If **Statistic >= Critical Value**: significant result, reject null hypothesis (H0), dependent.
#  - If **Statistic < Critical Value**: not significant result, fail to reject null hypothesis (H0), independent.
# 
# The *degrees of freedom* for the chi-squared distribution is calculated based on the size of the contingency table as (rows - 1) * (cols - 1).
# 
# In terms of a **p-value** and a chosen significance level **alpha**, the test can be interpreted as follows:
# 
#  - If **p-value <= alpha**: significant result, reject null hypothesis (H0), dependent.
#  - If **p-value > alpha**: not significant result, fail to reject null hypothesis (H0), independent.
#  
# For the test to be effective, at *least five observations are required in each cell of the contingency table*.
# 
# Next, let’s look at how we can calculate the chi-squared test.
# 
# Source: https://machinelearningmastery.com/chi-squared-test-for-machine-learning/


from scipy.stats import chi2
from scipy.stats import chi2_contingency


# The Pearson’s chi-squared test for independence can be calculated in Python using the *chi2_contingency()* *SciPy* function.
# 
# The function takes an array as input representing the contingency table for the two categorical variables. It returns the calculated statistic and p-value for interpretation as well as the calculated degrees of freedom and table of expected frequencies.


stat, p, dof, expected = chi2_contingency(ct)


stat, p, dof, expected


# ## Hypothesis
# For our test, we chose the followings hypothesis:
#  - **H0**: E[x] = m_i  for each group i, which means that there is no home-field advantage whitin the data we have observed.
#  - **H1**: E[x] != m_i  for at least one group i, which means that we are going to refuse H0 null Hypothesis and a home-field advantage is present whitin the data collected.
#  
# We can follow 2 way, each way will lead to the same result:
# 
# ### Percentiles Approach
# We can interpret the statistic by retrieving the critical value from the chi-squared distribution for the probability and number of degrees of freedom.
# 
# We choose a probability of 95%, suggesting that the finding of the test is quite likely given the assumption of the test that the variable is independent. If the statistic is less than or equal to the critical value, we can fail to reject this assumption, otherwise it can be rejected.


# interpret test-statistic
prob = 0.95
critical = chi2.ppf(prob, dof)
if abs(stat) >= critical:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


print('probability = %.3f, critical = %.3f, stat = %.3f' % (prob, critical, stat))


# With this approach, we see that we accept the null hypothesis H0: all the values belong to the same population, i.e. there is any kind of home-field advantage.


# ### P-Value Approach
# We can also interpret the p-value by comparing it to a chosen significance level, which would be 5%, calculated by inverting the 95% probability used in the critical value interpretation:


# interpret p-value
alpha = 1.0 - prob
if p <= alpha:
    print('Dependent (reject H0)')
else:
    print('Independent (fail to reject H0)')


print('significance = %.3f, p= %.3f' % (alpha, p))


# The result of this approach is the same of the percentiles: we accept the null hypothesis H0, which menas that all the values belong to the same population, i.e. there is any kind of home-field advantage. This phenomenon is not present within the selected five teams: 'Huddersfield Town': 1673, 'Southampton': 1619, 'Stoke City': 1639, 'Swansea City': 10531 e 'West Bromwich Albion': 1627. We cannot generalize this statistic just with just one test.
# 
# ## Iteration of Chi-Squared Test
# 
# We are going to replicate the test several times (n = 1000), picking randomly each time a subset of five teams. For this purpouse we have created the function pickingTeams.


picked = teams[teams.championshipId == 380].sample(n=5, frac=None, replace=False, weights=None, random_state=None, axis=None)


picked


dictTeams = { }
for i in range(len(picked)):
    # Adding a new key value pair
    dictTeams.update( {picked.iloc[i]['name'] : picked.iloc[i]['teamId']} )


dictTeams


dictTeams.items()


teamsList = []
for key,value in dictTeams.items():
        teamsList.append(value)


# type: it's the type of the team ('club' or 'national')
def pickingTeams(teams, teamType, championshipId, n_sample):
    dictTeams = { }
    teamsList = []
    picked = teams[ (teams.championshipId == championshipId) & (teams.type == teamType) ].sample(n=n_sample)

    for i in range(len(picked)):
        dictTeams.update( {picked.iloc[i]['name'] : picked.iloc[i]['teamId']} )

    for key,value in dictTeams.items():
            teamsList.append(value)
            
    return teamsList


# ## The Overall Test


def homeFieldAdvantageTest(n, championshipId, teams, teamType, table, n_sample, significance):
    countRejectH0 = 0
    countAcceptH0 = 0

    for i in range(n):
        lstTms = pickingTeams(teams, teamType, championshipId, n_sample)

        ctab = contingencyTable(lstTms, table)

        stat, p, dof, expected = chi2_contingency(ctab)

        # interpret p-value
        if p <= significance: countRejectH0 += 1
        else: countAcceptH0 += 1
    
    testOutcome = 'Percentage of rejections of H0  = %.3f\nPercentage of failing into reject of H0  = %.3f'
    return print(testOutcome % (countRejectH0/n, countAcceptH0/n))


homeFieldAdvantageTest(1000, 0, teams, 'club', tableEng, 5, 0.07)


# Assuming the null hypothesis H0 that there is no home-field advantage, within a simulation of 1000 times we see that **we reject H0 the 68% of the times**. In the Chi-Squared test processed we have relaxed the significance at the 7%, in this way we include in the rejection of H0 values which are close to the significance value of 5%.
# **We can conclude that there is a home-field advantage** within the matches analyzed.


# # [RQ3]


# ### Which teams have the youngest coaches?


# #### We need just name and wyId columns of teams in further commands


teamss = team[["name","teamId"]]
teamss


# #### By setting the wyId for index, it is easier to search with IDs


coach.set_index("wyId",inplace=True)


coach.head()


# #### We have some coaches that their currentTeamIds are 0, so we should search in matches(England) -> TeamId -> coachId 
# #### In order to find the teams with the coaches they had during the season


# #### Here we define 2 default dictionaries to fill with coachIds


coachesdic = defaultdict(lambda: [])
coachesdic2 = defaultdict(lambda: [])
coachData = dict(Match_England["teamsData"])
for key,value in coachData.items():
    alpha = value
    for key,value in alpha.items():
        coachesdic[key].append(value["coachId"]) 


# #### So we have a dict (coachesdic) which shows the coachIds for every weeks of matches of a specific team during the season


for key,value in coachesdic.items():
    alpha = set(value)
    value = list(alpha)
    value = [i for i in value if i != 0] 
    coachesdic2[key].append(value)


# #### Here we have English teamIds with the unique coachIds during the season (coachesdic2)


# #### And we dataframe the coachesdic2 and name the column CID(Coaches' Ids for each team)


df = pd.DataFrame(coachesdic2)
df_coach = df.T

df_coach.columns = ["CID"]

df_coach.head()


# #### So then we should choose the youngest coach for each team that had more than one coach


# #### For this, we use another defaultdic (age) for choosing the youngest one by their birthdate.
# #### But there was a problem with datasets: in a few weeks of the matches, one team had got a coach with id = 3782, and coachid= 3782 were not in coaches datasets, so we had to use exceptions.


age = defaultdict(lambda:[])
for i in range(20):
    for j in range(len(df_coach["CID"][i])):
        try:
            age[df_coach.index[i]].append(coach["birthDate"][df_coach["CID"][i][j]])
        except:
            pass  


df_coach.reset_index(inplace=True)


# #### Here we comparing the coach's birthDates with sorting them in a dataframe and picking the last row and append them in a list. 
# #### Then adding a new column with the lists that shows the index of the youngest one.


lis = []
for k,v in age.items():
    if k:
        dfk = pd.DataFrame(v,index=[i for i in range(len(v))])
        dfk.sort_values(0,inplace=True)
        young = dfk[0][-1:]
        answer = int(str(young).split()[0])
        lis.append(answer)


df_coach["Yind"] = lis


# #### Renaming the columns...


df_coach.columns = ["teamid","coachid","Yind"]


# #### As said before there is no coachId with id=3782 so we change the index of "Yind" where we have 3782


df_coach["Yind"][17] = "1"


df_coach["Yind"][17]


# #### And in lis2 we have all the youngest one's birthdays and add them in a new col. in df_coach


lis2 = []
for i in range(len(df_coach)):
        lis2.append(coach["birthDate"][df_coach["coachid"][i][df_coach["Yind"][i]]])


df_coach["birthDate"] = lis2


# #### Then we change the coachid column with lis3(inorder to have the unique coachid)


lis3 = []
for i in range(len(df_coach)):
        lis3.append(df_coach["coachid"][i][df_coach["Yind"][i]])
        


df_coach.drop("coachid",axis=1,inplace=True)
df_coach["coachid"] = lis3


# #### Also we no more need the "Yind" column


df_coach.drop("Yind",axis=1,inplace=True)


Eng = pd.DataFrame(eng)


Eng.columns = ["teamid"]


# #### And merge it with teams to have the teamId and their name(Only the English ones)


Eng = pd.merge(Eng,teamss,left_on="teamid",right_on="teamId")


Eng


# #### We can drop the wyId because we don't need it


Eng.drop("teamId",axis=1,inplace=True)


Eng.sort_values("teamid") #Not necessary 


df_coach.sort_values("teamid") #Not necessary 


# #### We numeric the "teamId" to be merged, so we change it into numeric values


df_coach["teamid"] = pd.to_numeric(df_coach["teamid"])


# #### Then we merge df_coach with Eng


df_coach = pd.merge(df_coach ,Eng,left_on="teamid",right_on="teamid")


df_coach.head()


df_coach.sort_values("birthDate",ascending=False,inplace=True)


# #### Now we have a sorted dataframe that the top 10 rows show the youngest coaches


youngestdf = pd.DataFrame
youngestdf = df_coach.iloc[:10]
youngestdf.head()


youngestdf = youngestdf[["name","coachid"]]


youngestdf = youngestdf.reset_index()


youngestdf.drop("index", axis=1,inplace=True)


# Also we no more need "coachId" column # Not necessary


youngestdf.drop("coachid", axis=1,inplace=True)


# ### The #10 teams with the youngest coaches
# 


youngestdf


# #### Now we calculate the ages and showing the their distribution with Boxplot


df_coach = df_coach.reset_index()

df_coach.drop("index",axis=1,inplace=True)


df_coach.head()


coaches = coach.reset_index()


# #### Calculating the ages


age = []
for i in range(len(df_coach)):
    born = list(map(int, df_coach.birthDate[i].split("-")))
    today = date.today()
    age.append(today.year - born[0])

df_coach['age'] = age


df_coach.head()


# #### For boxplot we just need the age col.


dfage = df_coach.drop(["teamid","birthDate","name","coachid"],axis=1)


dfage


# #### And Finally the Boxplot


color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange','medians': 'DarkBlue', 'caps': 'Gray'}
dfage.plot.box(color = color,
              sym='ro', figsize= (7,7))



# # [RQ4]


# ### Find the top 10 players with the highest ratio between completed passes and attempted passes


eng_playersP= eng_players.copy()


# #### We add two colums that counts the completed and attempted passes for every player in the Premier Legue. In the beginning the count is initialized at all 0, later, with the dataframe of ONLY passes, we can count.


eng_playersP["completedPasses"]= [0 for _ in range(len(eng_players))]
eng_playersP["AttemptedPasses"]= [0 for _ in range(len(eng_players))]


eng_playersP.head()


# Only events whose event name is Pass
df_passes = event_England.loc[event_England['eventName']=="Pass"]

df_passes.head()


#to adjust the indexing

pd.DataFrame.reset_index(df_passes, inplace= True)

df_passes.drop(columns=['index'])


# #### We count the completed passes and the attempted passes by iterating on the passes_dataframe. If a pass has tag 1801 and 1401 is considered attempted, because it was accurate but was intercepted, otherwise is a pass has only tag 1801 is considered completed (the other tags that come with 1801 are the description of the passes, so they are not relevant in our analysis


for i in range(len(df_passes)):
    if ( 1801 in df_passes.tagsList[i] and 1401 not in df_passes.tagsList[i]):
        eng_playersP.loc[eng_players.playerId ==df_passes.playerId[i], "completedPasses"]+=1


for i in range(len(df_passes)):
    if ( 1801 in df_passes.tagsList[i]):
        eng_playersP.loc[eng_players.playerId ==df_passes.playerId[i], "AttemptedPasses"]+=1


# This is what we've got 
eng_playersP


#thresholdP (passes) chosen = the 2nd quartile on all attempted passsages, in this way we keep only the players who have done a significant number of passages

thresholdP= eng_players['AttemptedPasses'].median()
thresholdP


# We are not interested in what is under the threshold.

# So we create an index of all the rows that have the number of Attemped Passes under our threshold.
indexNamesP = eng_players[ eng_playersP['AttemptedPasses'] < thresholdP ].index
 
# And delete these row indexes from dataFrame
eng_playersP.drop(indexNamesP , inplace=True)


eng_playersP


# We calculate the ratio between completed passes and attempted passes

eng_playersP['complPass/attemPass']=eng_playersP["completedPasses"]/eng_playersP['AttemptedPasses']

eng_playersP


# ### Finally, the top 10 players with the hightest ration between Completed Passes and Attempted Passes are:


eng_playersP.nlargest(10, "complPass/attemPass")


# # [RQ5]


# ### Does being a tall player mean winning more air duels? 


# #### As in RQ4, we add two columns to keep counter of the Attempted and Compleated air duels for each players in the Premier Legue


eng_playersD= eng_players.copy()


eng_playersD["CompletedAirDuels"]=[0 for _ in range(len(eng_players))]
eng_playersD["AttemptedAirDuels"]= [0 for _ in range(len(eng_players))]


# From all the events in all the matches, we extract only those whose subNameEvent is "Air duels"

df_AirDuels= event_England[(event_England.subEventName == 'Air duel')]

df_AirDuels


# Reindexing
pd.DataFrame.reset_index(df_AirDuels, inplace= True)

df_AirDuels.drop(columns=['index'])


# #### And now we update the counter of the completed Air duels and the Attempted Air Duels.


# #### We iterate on all the airduels, and at the i-th AirDuel, if it has 1801 (accurate event) and tag 703 (duel won), the event is considerated Completed, else if it has only 1801 tag is considered attempted. The other tags that come with 1801 are mainly 701 (lost), 702 (neutral), and tags about the description of the duel, and they are not relevant for our analysis.


for i in range(len(df_AirDuels)):
    if ( 1801 in df_AirDuels.tagsList[i] and 703 in df_AirDuels.tagsList[i]):
        eng_playersD.loc[eng_playersD.playerId ==df_AirDuels.playerId[i], "CompletedAirDuels"]+=1


for i in range(len(df_AirDuels)):
    if ( 1801 in df_AirDuels.tagsList[i]):
        eng_playersD.loc[eng_players.playerId ==df_AirDuels.playerId[i], "AttemptedAirDuels"]+=1


eng_playersD


thresholdAD= eng_playersD['AttemptedAirDuels'].median()
thresholdAD


# We are not interested in what is under the threshold.
# So we create an index of all the rows that have the number of Attemped AirDuels under a certain threshold.
indexNamesD = eng_playersD[ eng_playersD['AttemptedAirDuels'] < thresholdAD ].index
 
# And delete these row indexes from dataFrame
eng_playersD.drop(indexNamesD , inplace=True)


eng_playersD["complAirDuels/attemAirDuels"]= eng_playersD["CompletedAirDuels"]/eng_playersD["AttemptedAirDuels"]


eng_playersD


sns.set_style('darkgrid')
plt.figure(figsize=(15,5))
sns.stripplot(x="height",y="complAirDuels/attemAirDuels",data=eng_playersD,size=8,jitter=True,dodge= True)
#sns.set_context(context="notebook",font_scale=1)


# their distributions and dependecies with regression
sns.jointplot(x="height",y="complAirDuels/attemAirDuels",data=eng_playersD,kind='reg',height=8)



# # [RQ6]


# ### Our goal for this exercise is to produce a table with the players who have done more asists than others


eng_playersA= eng_players.copy()


eng_playersA["Assistcount"]= [0 for _ in range(len(eng_players))]
eng_playersA


for i in range(len(event_England)):
    if ( 301 in event_England.tagsList[i]):
        eng_playersA.loc[eng_players.playerId ==event_England.playerId[i], "Assistcount"]+=1


eng_playersA.sort_values("Assistcount",ascending=False,inplace=True)
eng_playersA


Assistdf = eng_playersA[0:11]
Assistdf


Assistdf.drop(columns={"middleName","currentTeamId","playerId", "height"},inplace=True)


#This is the table we were looking for!!

Assistdf


# # [CRQ1]


# ### What are the time slots of the match with more goals?


# #### Here make a dataframe containing for every goal: its Tagslist, its playerId, its teamId, its matchPeriod, its event second and its event minute 


goals=[]

for i in range(len(event_England)):
    if (101 in event_England.tagsList[i] and 1801 in event_England.tagsList[i]):
         goals.append([event_England.tagsList[i], event_England.playerId[i], event_England.teamId[i], event_England.matchPeriod[i], event_England.eventSec[i], event_England.eventMin[i]])


Goals= pd.DataFrame(goals)
Goals.rename(columns = {0:' tagsList', 1:'playerId', 2:'teamId', 3:'matchPeriod', 4:'eventSec', 5 :'eventMin'}, inplace = True)


# And this is what we get
Goals


# We add a new column which show the minute of the goal in an interval of 90' (not of 45')
TotalEventMin=[]

for i in range(len(Goals)):
    if Goals.matchPeriod[i]=="2H":
        TotalEventMin.append(Goals.eventMin[i]+45)
    else:
        TotalEventMin.append(Goals.eventMin[i])


Goals["totalEventMin"]=TotalEventMin

Goals.head()


# #### We count in witch interval falls each goal, according to the following table


#interval= 1 ---> 0'-9'
#interval= 2 ---> 9'-18'
#interval= 3 ---> 18'-27'
#interval= 4 ---> 27'-36'
#interval= 5 ---> 36'-45'
#interval= "45+" ---> for the extra time in the 1st period of the match
#interval= 6 ---> 45'-54'
#interval= 7 ---> 54'-63'
#interval= 8 ---> 63'-72'
#interval= 9 ---> 72'-81'
#interval= 10 ---> 81'-90'
#interval= "90+" ---> for the extra time in the 2nd period of the match

interval=[] #the interval of 9 min in which belong every goal


for i in range(len(Goals)):
    if (Goals.matchPeriod[i] == "1H" and Goals.eventMin[i]>45): 
        interval.append("45+")
    elif (Goals.matchPeriod[i] == "2H" and Goals.eventMin[i]>45):
        interval.append('90+')
    else:
        for j in range(1, 11):
            if  (Goals.totalEventMin[i]/ j <= 9)== True :
                interval.append( j)
                break



Goals["interval"]= interval


Goals.head()


# ## i. The barplot with the absolute frequency of goals in all the time slots.


interv=[1,2,3,4,5,"45+",6,7,8,9,10,"90+"]
frequency_g=[]


for el in interv:
    frequency_g.append(len(Goals[Goals.interval== el]))


#frequency of the goals in each interval of time
frequency_g


# The Plot

itvs=["0-9 min", "9-18 min","18-27 min","27-36 min","36-45 min","45+","45-54 min","54-63 min","63-72 min","72-81 min","81-90 min", "90+"]
y_pos= [i for i in range(len(interv))]

plt.style.use('seaborn-darkgrid')

plt.figure(figsize=(13,7)) 
plt.bar(y_pos, frequency_g,color="powderblue",edgecolor='green')
plt.xticks(y_pos, itvs)
plt.ylabel("Frequency of goals",  fontsize = 20)
plt.title("Distribution of goals in the Premier Legue 2017/2018", loc = 'center', fontsize = 30, fontweight = 3, color = 'lightseagreen')
plt.show()


# ## ii. Find the top 10 teams that score the most in the interval "81-90".


# Here we count how many goals has done each team in the interval 81-90 min on the whole season

goalperteam=[]

for team in eng:
    goalperteam.append([team, len(Goals[(Goals.teamId== team) &(Goals.interval==10)])])


goalperteam_df= pd.DataFrame(goalperteam, columns=["teamId", "number of goal between 81th and 90th"])


# This is the top 10 required

goalperteam_df.nlargest(10,'number of goal between 81th and 90th')


# ## iii. Show if there are players that were able to score at least one goal in 8 different intervals.


# Here for every player ho scored a goal, in a given interval we put an 1 if he has done at least one goal , otherwise we put 0

goalperplayer ={}

for player in Goals.playerId:
            goalperplayer2[player]= [1 if len(Goals[(Goals.playerId == player) &(Goals.interval==el)])>0 else 0 for el in interv]



goalperplayer_df= pd.DataFrame(goalperplayer2).T

goalperplayer_df


# If we sum on each row (on ach player) we can see in how many interval they have done (at least) one goal
goalperplayer_df['tot']=goalperplayer_df.sum(axis=1)


goalperplayer_df


# So since we are looking for the player who scored (at least) one goal in 8 different intervals 

goalperplayer_df[goalperplayer_df.tot == 8 ]


# # [CRQ2]
# ## Visualize movements and passes on the pitch!
# Here we try to focus our attention on the zones that a player covers during a match. For each event, we have a pair of coordinates, that are respectively the starting and ending point of that event.
# 
# It can be helpful to follow this link: https://towardsdatascience.com/advanced-sports-visualization-with-pandas-matplotlib-and-seaborn-9c16df80a81b
# 
# Knowing all the different positions where events happen, let us be able to create different types of visualizations:
# 
# ## Considering only the match Barcelona - Real Madrid played on the 6 May 2018:
# 
# Visualize with a **heatmap** the zones where **Cristiano Ronaldo** was more active.
# The events to be considered are: **passes**, **shoots**, **duels**, **free kicks**.
# 
# Compare his map with the one of **Lionel Messi**. Comment the results and point out the main differences (we are not looking for deep and technique analysis, just show us if there are some clear differences between the 2 plots).


matches_Spain[matches_Spain.label.str.contains('Barcelona') & matches_Spain.label.str.contains('Real Madrid')]
# MatchId: 2565907


# Events of Barcelona - Real Madrid, 6 May 2018, Camp Nou
events_FCB_RMA = events_Spain[events_Spain.matchId == 2565907 ]
events_FCB_RMA.head()


players[ (players.playerId == 3359 ) | (players.playerId == 3322 ) ]
# Messi 3359
# Ronaldo 3322 (belongs to Juventus Football Club)


events_Ronaldo = events_FCB_RMA.loc[events_FCB_RMA['playerId'] == 3322]
events_Messi = events_FCB_RMA.loc[events_FCB_RMA['playerId'] == 3359]


def draw_pitch(ax):
    # focus on only half of the pitch
    #Pitch Outline & Centre Line
    Pitch = plt.Rectangle([0,0], width = 120, height = 80, fill = False)
    #Left, Right Penalty Area and midline
    LeftPenalty = plt.Rectangle([0,22.3], width = 14.6, height = 35.3, fill = False)
    RightPenalty = plt.Rectangle([105.4,22.3], width = 14.6, height = 35.3, fill = False)
    midline = ConnectionPatch([60,0], [60,80], "data", "data")

    #Left, Right 6-yard Box
    LeftSixYard = plt.Rectangle([0,32], width = 4.9, height = 16, fill = False)
    RightSixYard = plt.Rectangle([115.1,32], width = 4.9, height = 16, fill = False)

    #Prepare Circles
    centreCircle = plt.Circle((60,40),8.1,color="black", fill = False)
    centreSpot = plt.Circle((60,40),0.71,color="black")

    #Penalty spots and Arcs around penalty boxes
    leftPenSpot = plt.Circle((9.7,40),0.71,color="black")
    rightPenSpot = plt.Circle((110.3,40),0.71,color="black")
    leftArc = Arc((9.7,40),height=16.2,width=16.2,angle=0,theta1=310,theta2=50,color="black")
    rightArc = Arc((110.3,40),height=16.2,width=16.2,angle=0,theta1=130,theta2=230,color="black")

    # Merging all the elements togheter
    element = [Pitch, LeftPenalty, RightPenalty,
               midline, LeftSixYard, RightSixYard, centreCircle,
               centreSpot, rightPenSpot, leftPenSpot,leftArc, rightArc]
    for i in element:
        ax.add_patch(i)


fig=plt.figure() #set up the figures
fig.set_size_inches(7, 5)
ax=fig.add_subplot(1,1,1)
draw_pitch(ax) #overlay our different objects on the pitch
plt.ylim(-2, 82)
plt.xlim(-2, 122)
#plt.axis('off')
plt.show()


events_Ronaldo = events_FCB_RMA.loc[events_FCB_RMA['playerId'] == 3322]
events_Messi = events_FCB_RMA.loc[events_FCB_RMA['playerId'] == 3359]


# Ronaldo's Events to consider:
events_Ronaldo['eventName'].unique()


# Messi's Events to consider:
events_Messi['eventName'].unique()


# Passes
ronaldo_passes = events_Ronaldo[(events_Ronaldo['eventName'] == "Pass")]
messi_passes = events_Messi[(events_Messi['eventName'] == "Pass")]

# Duels
ronaldo_duels = events_Ronaldo[(events_Ronaldo['eventName'] == "Duel")]
messi_duels = events_Messi[(events_Messi['eventName'] == "Duel")]

# Free Kicks
ronaldo_freeKicks = events_Ronaldo[(events_Ronaldo['eventName'] == "Free Kick")]
messi_freeKicks = events_Messi[(events_Messi['eventName'] == "Free Kick")]

# Shots
ronaldo_shots = events_Ronaldo[(events_Ronaldo['eventName'] == "Shot")]
messi_shots = events_Messi[(events_Messi['eventName'] == "Shot")]

# Shots
ronaldo_goals = events_Ronaldo[(events_Ronaldo['eventName'] == "Shot") & (events_Ronaldo.tagsStr.str.contains('S101E')) ]
messi_goals = events_Messi[(events_Messi['eventName'] == "Shot") & (events_Messi.tagsStr.str.contains('S101E'))]



ronaldo_passes.columns


fig, ax = plt.subplots()
fig.set_size_inches(7, 5)
ax=fig.add_subplot(1,1,1)
draw_pitch(ax) #overlay our different objects on the pitch
plt.ylim(-2, 82)
plt.xlim(-2, 122)
#plt.axis('off')

x_coord = [i*1.2 for i in ronaldo_duels["startPosX"]]
y_coord = [i*0.8 for i in ronaldo_duels["startPosY"]]

#shades: give us the heat map we desire
# n_levels: draw more lines, the larger n, the more blurry it looks
sns.kdeplot(x_coord, y_coord, shade = "True", color = "green", gridsize=100, ax=None, n_levels = 30)
plt.axis('off')
plt.show()



# Arrows
fig=plt.figure()
fig.set_size_inches(7, 5)
ax=fig.add_subplot(1,1,1)
draw_pitch(ax)
plt.axis('off')
    
for i in range(len(ronaldo_passes)):
    # annotate draw an arrow from a startPos to endPos
    ax.annotate("", xy = (ronaldo_passes.iloc[i]['endPosX']*1.2, ronaldo_passes.iloc[i]['endPosY']*1.2), xycoords = 'data',
               xytext = (ronaldo_passes.iloc[i]['startPosX']*1.2, ronaldo_passes.iloc[i]['startPosY']*1.2), textcoords = 'data',
               arrowprops = dict(arrowstyle="->",connectionstyle="arc3", color = "orangered"),)

x_coord = [i*1.2 for i in ronaldo_duels["startPosX"]]
y_coord = [i*0.8 for i in ronaldo_duels["startPosY"]]

#shades: give us the heat map we desire
# n_levels: draw more lines, the larger n, the more blurry it looks
sns.kdeplot(x_coord, y_coord, shade = "True", color = "green", n_levels = 30)

plt.ylim(-2, 82)
plt.xlim(-2, 122)
plt.show()



shots = events_Ronaldo[(events_Ronaldo['eventName'] == "Shot")]
shots.tagsStr.str.contains('S101E').iloc[1]


shots.iloc[0]


[yCord for j, yCord in enumerate(shots["startPosY"]) if 101 in shots.tagsList.iloc[j]]


def activitiesHeatmap(events_match, playerId, typeHeatMap='simple'): #3322 #3359
    
    events_player = events_match.loc[events_match['playerId'] == playerId]
    
    actions = events_player[(events_player['eventName'] == "Pass") | 
                            (events_player['eventName'] == "Duel") |
                            (events_player['eventName'] == "Shot") |
                            (events_player['eventName'] == "Free Kick")]
    
    fig=plt.figure()
    fig.set_size_inches(7, 5)
    ax=fig.add_subplot(1,1,1)
    draw_pitch(ax)
    plt.axis('off')
    
    x_coord = [i*1.2 for i in actions["startPosX"]]
    y_coord = [i*0.8 for i in actions["startPosY"]]
    
    if (typeHeatMap == 'passes') | (typeHeatMap == 'aggregate'):
        passes = events_player[(events_player['eventName'] == "Pass")]
        for i in range(len(passes)):
            color = 'blue' if 1801 in passes.tagsList.iloc[i] else 'red'#red if pass is inaccurate
            # annotate draw an arrow from a startPos to endPos
            ax.annotate("", xy = (passes.iloc[i]['endPosX']*1.2, passes.iloc[i]['endPosY']*0.8), xycoords = 'data',
                       xytext = (passes.iloc[i]['startPosX']*1.2, passes.iloc[i]['startPosY']*0.8), textcoords = 'data',
                       arrowprops = dict(arrowstyle="->",connectionstyle="arc3", color = color),)

    sns.kdeplot(x_coord, y_coord, shade = "True", color = "green", n_levels = 30)
    
    if typeHeatMap == 'aggregate':
        typeHeatMap = 'shots'
        
    if (typeHeatMap == 'shots') | (typeHeatMap == 'aggregate'):
        shots = events_player[(events_player['eventName'] == "Shot")]
        x_shot = [j*1.2 for j in shots["startPosX"]]
        y_shot = [j*0.8 for j in shots["startPosY"]]
        x_gol = [xCord*1.2 for z, xCord in enumerate(shots["startPosX"]) if 101 in shots.tagsList.iloc[z]]
        y_gol = [yCord*0.8 for z, yCord in enumerate(shots["startPosY"]) if 101 in shots.tagsList.iloc[z]]
        ax.scatter(x_shot, y_shot, c = "blue", label = 'Shot')
        ax.scatter(x_gol, y_gol, c = "gold",label = 'Goal')
        plt.legend(loc='upper right')
    
    plt.ylim(-2, 82)
    plt.xlim(-2, 122)

    return  plt.show()


# Cristiano Ronaldo
activitiesHeatmap(events_FCB_RMA, 3322, typeHeatMap='aggregate')


# Leo Messi
activitiesHeatmap(events_FCB_RMA, 3359, typeHeatMap='aggregate')


# #### In the aggregated statistics we see the main characteristics of these two phenomenons of the modern foootball: thanks to the big and strong body,  *Cristiano Ronaldo* could cover all the ground, the movents he does all over the ground are foundamental, he has a global impact on the match. Passing the ball  is not one of the most  important characteristics of him: close to the door, he prefers to shot. 
# #### One important characteristic we could see from these observations is that he is  always accurate in short passages ( the blue arrows. A red arrow is a non accurate pass).
# #### He is an area hunter: his presence in the area is costant, and thanks to this, he scored one goal, being at the right moment in the rigth place.
# #### For all these characteristics, Cristiano Ronaldo represent one of the most strong player of the entire century.
# #### *Leo Messi* is as dangerous as Cristiano, but his characteristics are completely  different: he cover a precise area in the middle of the ground, and he moves  just when he believes is necessary. He handles a lot of balls: Leo has the capability of doing precise passes inside the opposite team area, from each  part of the ground.
# #### He has all the best characteristics of a midfielder and a striker at the same time, but he is not famous for being powerful and resilient in running for a long time.
# #### This is clear analyzing data: compared to Cristiano Ronaldo, he covers just the half of the ground. Despite this, Messi has a good balance in passing and shooting when he reaches the area, and all his actions are dangerous, he is impredictable. Thanks to all this, Messi scored one goal at the beginning of the first half.
# 


# ## Considering only the **match Juventus - Napoli** played on the 22 April 2018:
# Visualize with **arrows** the starting point and ending point of each pass done during the match by **Jorginho** and **Miralem Pjanic**. Is there a huge difference between the map with all the passes done and the one with only accurate passes? Comment the results and point out the main differences.


matches_Italy = functions.cleaningMatches(match_Italy)
events_Italy = functions.CleanEngEvents(event_Italy)


# Events of Juventus - Napoli, 22 April 2018, Juventus Stadium
matches_Italy[matches_Italy.label.str.contains('Juventus') & matches_Italy.label.str.contains('Napoli')]
# MatchId: 2576295


events_JUV_NAP = events_Italy[events_Italy.matchId == 2576295 ]


events_JUV_NAP.head()


players[ (players.playerId == 20443 ) | (players.playerId == 21315 ) ]
# Miralem Pjanic 20443
# Jorghinho 21315


# Miralem Pjanic
activitiesHeatmap(events_JUV_NAP, 20443, typeHeatMap='passes')


# Jorginho
activitiesHeatmap(events_JUV_NAP, 21315, typeHeatMap='passes')


# #### From these heatmaps, we can see how much strong could be *Miralem Pjanic*: he his a complete midfielder, he can do all the different phases in the middle of the ground. His percentage of completed passes is one of the highest between the midfielders. He covers all the ground thanks to his presence and his speed.
# #### *Jorginho* has a different way of covering the midfield: he prefers to cover a short area of the ground, and in this area he is a fluidificator in doing short passes with success. He tried to invent interesting actions for the forwards, but any of his passes reached them with success, probabily thanks to the Juventus defense, one of the strongest of the Italian Championship. We have seen also here two different ways for being a great champion.





