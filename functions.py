import pandas as pd

def CleanEngEvents(event_England):
    positionDF = pd.DataFrame(event_England.positions.tolist(), columns=['startPos','endPos'] )
    tagsDF = pd.DataFrame(event_England.tags.tolist())
    startPosX=[]
    startPosY=[]
    endPosX=[]
    endPosY=[]
    tagStr=[]
    tagList=[]

    for i in range(len(positionDF)):
        #POSITIONS
        try:
            startPosX.append(positionDF.startPos[i].get("x"))
            startPosY.append(positionDF.startPos[i].get("y"))
            endPosX.append(positionDF.endPos[i].get("x") )
            endPosY.append(positionDF.endPos[i].get("y"))
        except:
            # Foul: Late Foul Cards, Time Lost Foul and Protest has no endPos!
            # In this way we create a point, setting endPos = startPos
            endPosX.append(positionDF.startPos[i].get("x"))
            endPosY.append(positionDF.startPos[i].get("y"))
        #TAGS
        ts    = []
        tlst = []
        for j in range(len(tagsDF.columns)):
            if tagsDF[j][i] is not None:
                ts.append('S' + str(tagsDF[j][i].get("id")).strip() + 'E')
                tlst.append(tagsDF[j][i].get("id"))
        tagStr.append(",".join(ts))
        tagList.append(tlst)

    #we add the columns od the positions and the tags as we want them
    event_England['startPosX'] = startPosX
    event_England['startPosY'] = startPosY
    event_England['endPosX'] = endPosX
    event_England['endPosY'] = endPosY
    event_England['tagsStr'] = tagStr
    event_England['tagsList'] = tagList

    # Dropping Columns we don't need
    event_England.drop(columns={"positions","tags"}, inplace=True)

    # Creating the column Minute of the Event
    event_England['eventMin'] = event_England['eventSec']//60
    return event_England

def cleaningMatches(matches):
    # Let's going to complete the informations about the MATCHES
    homeTeamId = []
    homeScore  = []
    homePoints = []
    homeWin    = []
    homeDraw   = []
    homeLoss   = []
    awayTeamId = []
    awayScore  = []
    awayPoints = []
    awayWin    = []
    awayDraw   = []
    awayLoss   = []

    # Retrieving the important data into Lists
    for i in range(len(matches)):
        lst = list(set(map(int, matches['teamsData'][i].keys())))
        for teamPoints in lst:
            if(matches['teamsData'][i][str(teamPoints)]['side'] == 'home'):
                homeTeamId.append(matches['teamsData'][i][str(teamPoints)]['teamId'])
                homeScore.append(matches['teamsData'][i][str(teamPoints)]['score'])
            else:
                awayTeamId.append(matches['teamsData'][i][str(teamPoints)]['teamId'])
                awayScore.append(matches['teamsData'][i][str(teamPoints)]['score'])
        if(homeScore[i] > awayScore[i]):
            homePoints.append(3)
            homeWin.append(1)
            homeDraw.append(0)
            homeLoss.append(0)
            awayPoints.append(0)
            awayWin.append(0)
            awayDraw.append(0)
            awayLoss.append(1)
        elif(homeScore[i] < awayScore[i]):
            homePoints.append(0)
            homeWin.append(0)
            homeDraw.append(0)
            homeLoss.append(1)
            awayPoints.append(3)
            awayWin.append(1)
            awayDraw.append(0)
            awayLoss.append(0)
        else:
            homePoints.append(1)
            homeWin.append(0)
            homeDraw.append(1)
            homeLoss.append(0)
            awayPoints.append(1)
            awayWin.append(0)
            awayDraw.append(1)
            awayLoss.append(0)

    # UPDATING TABLE
    # Home Result
    matches['homeTeamId'] = pd.Series(homeTeamId)
    matches['homeScore']  = pd.Series(homeScore)

    # Away Result
    matches['awayScore']  = pd.Series(awayScore)
    matches['awayTeamId'] = pd.Series(awayTeamId)

    # Event Table
    matches['homeWin']    = pd.Series(homeWin)
    matches['homeDraw']   = pd.Series(homeDraw)
    matches['homeLoss']   = pd.Series(homeLoss)
    matches['awayWin']    = pd.Series(awayWin)
    matches['awayDraw']   = pd.Series(awayDraw)
    matches['awayLoss']   = pd.Series(awayLoss)

    # Points
    matches['homePoints'] = pd.Series(homePoints)
    matches['awayPoints'] = pd.Series(awayPoints)

    #Sorting
    matches.sort_values("gameweek", inplace=True)

    #Reindexing
    matches.reset_index(inplace=True)

    # Renaming columns
    matches.rename(columns={"wyId": "matchId"}, inplace= True)

    #Dropping columns
    matches.drop(columns=['status', 'roundId', 'seasonId', 'referees', 'duration', 'date', 'venue', 'index'], inplace=True)

    return matches

def CountingPoints(DF, engTableNGW):
# We make a dictionary for the weeks and transform it into a dataframe.
# This will be concatenate to the scores dataset and used ad x axis in the final plot
    week={"gameweek": [i for i in range(0,39)]}
    week= pd.DataFrame(week)

    #We iterate on every match (j) and every week (i).
#    We extract the players of the j-th match
#    If we have a winner at the j-th position of the i=th week,
#    the score is updated (into the dict engTableNGW created before) from the score of the previous week+3
#    (also the score of the looser is updated: his score at the i-th week is the same of the (i-1)th week);
#    Else if we don't have a winner (winner== 0), we update both scores: their score at the i-th week are the
#    same of the (i-1)th week (into the dict engTableNGW created before).
    for i in range(1,39):
        for j in range(len(DF)):
            lst = list(set(map(int, DF['teamsData'][j].keys())))
            if (DF['winner'][j] > 0 and DF['gameweek'][j] == i ):
                for teamPoints in lst:
                    if DF['winner'][j] == teamPoints:
                        engTableNGW[teamPoints][i] = engTableNGW[teamPoints][i-1] + 3
                    else:
                        engTableNGW[teamPoints][i] = engTableNGW[teamPoints][i-1]

            elif (DF['winner'][j] == 0 and DF['gameweek'][j] == i ):
                engTableNGW[lst[0]][i] = engTableNGW[lst[0]][i-1] + 1
                engTableNGW[lst[1]][i] = engTableNGW[lst[1]][i-1] + 1

    # This creates a Dataframe from the dictionary Eng_teamNGW (with all the scores updated week per week),
    #and concatenate it to the Week dataframe created before
    scores = pd.DataFrame(engTableNGW)
    scores= pd.concat([week, scores], axis= 1)
    return scores


def cleaningTeams(teams):
    # Unnesting field into the DATAFRAME TEAMS
    areaName = []
    areaId = []
    areaCode = []

    for i in range(len(teams)):
        areaName.append(teams.area[i]['name'])
        #areaId.append(teams.area[i]['id']) # it is a mixture of strings and integer, series could have different types inside
        # national has this field as an integer
        #if we leave areaId in this way, .sort_values("championshipId") doesn't work
        areaId.append(int(teams.area[i]['id']))
        areaCode.append(teams.area[i]['alpha3code'])

    teams['areaName'] = pd.Series(areaName)
    teams['championshipId'] = pd.Series(areaId)
    teams['areaCode'] = pd.Series(areaCode)

    # Renaming columns
    teams.rename(columns={"wyId": "teamId"}, inplace= True)

    #Dropping columns
    teams.drop(columns=['area'], inplace=True)

    return teams

def cleaningPlayers(players):
    # Unnesting field into the DATAFRAME TEAMS
    roleId        = []
    roleName      = []
    birthArea     = []
    passportArea  = []

    for i in range(len(players)):
        roleName.append(players.role[i]['name'])
        roleId.append(players.role[i]['code3'])
        birthArea.append(players.birthArea[i]['alpha3code'])
        passportArea.append(players.passportArea[i]['alpha3code'])

    players['roleId'] = pd.Series(roleId)
    players['roleName'] = pd.Series(roleName)
    players['birthArea'] = pd.Series(birthArea)
    players['passportArea'] = pd.Series(passportArea)

    # Renaming columns
    players.rename(columns={"wyId": "playerId"}, inplace= True)

    #Dropping columns
    players.drop(columns=['role'], inplace=True)

    return players


def creatingTable(teams, matches):
    # Retrieving Home/Away Statistics & Points
    homeMatches = pd.merge(teams.filter(['name', 'teamId', 'championshipId']), matches, left_on='teamId', right_on='homeTeamId', how='inner')
    awayMatches = pd.merge(teams.filter(['name', 'teamId', 'championshipId']), matches, left_on='teamId', right_on='awayTeamId', how='inner')

    # The subtable of Home and Away matches
    hm = homeMatches[['teamId', 'name','homeWin','homeDraw','homeLoss','homePoints' ]].groupby(['teamId','name']).sum()
    am = awayMatches[['teamId', 'name','awayWin','awayDraw','awayLoss','awayPoints' ]].groupby(['teamId','name']).sum()

    # The Final Table- innerjoin between hm and am, retrieving all the relevant information
    table = pd.merge(hm, am, on='teamId' )

    # Calculating Total Points
    table['totalPoints'] = table['homePoints'] + table['awayPoints']
    table['totalWin']    = table['homeWin']    + table['awayWin']
    table['totalDraw']   = table['homeDraw']   + table['awayDraw']
    table['totalLoss']   = table['homeLoss']   + table['awayLoss']

    # Joining teams' names
    table = pd.merge(teams, table, left_on='teamId', right_on='teamId', how='inner')[['teamId','name','totalWin','totalDraw','totalLoss','homeWin','homeDraw','homeLoss','awayWin','awayDraw','awayLoss','homePoints','awayPoints','totalPoints']]

    # Sorting the table
    table.sort_values("totalPoints", ascending=False, inplace=True)

    #Reindexing
    table.reset_index(inplace=True)

    #Dropping columns
    table.drop(columns=['index'], inplace=True)

    return table


def singleContingencyTable(tmId, table):
    # This function creates the Contingency Table for one team
    # INPUT: One teamId
    # OUTPUT: DataFrame single Contingency table
    df1 = table[table.teamId == tmId][['teamId','name','homeWin','homeDraw','homeLoss']]
    df2 = table[table.teamId == tmId][['teamId','name','awayWin','awayDraw','awayLoss']]
    df1.rename(columns={"homeWin": "win", "homeDraw": "draw", "homeLoss" : "loss"}, inplace=True)
    df2.rename(columns={"awayWin": "win", "awayDraw": "draw", "awayLoss" : "loss"}, inplace=True)
    df1['typeVenue'] = ['home']
    df2['typeVenue'] = ['away']
    return pd.concat([df1,df2], axis=0)
