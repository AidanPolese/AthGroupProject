import csv
from pprint import pprint as pp
import numpy as np

import jellyfish
import lda
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

CUSTOM_SPECIAL_CHARACTERS = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '-', '=', '+', '[', ']', '{', '}', '\\', '|', ';', ':', '\'', '\"', ',', '<', '.', '>', '?', '/', '`', '~']
SIMPLIFIED_GROUPS = {'Podcasts': 994, 'House Music': 2278, 'Throwback Music': 973, 'Pop Music': 2171, 'Indie Music': 688, 'Punk Music': 300, 'Silence': 801, 'Sports Programming': 286, 'Spotify Playlists': 355, 'npr': 235}


def checkSimilarInterests(interests, interestSimilarities, counter):
    # * a recursive funtion that is used to categorize similar listening interests between people using JWD comparison
    if len(interests) == 0:
        return
    # arbitrarilly choose the main interest to compare to
    mainInterest = interests[0]
    interestSimilarities[mainInterest] = [mainInterest]
    # check all people's interests to see which are similar to the main interest selected
    for interestIndex in range(1, len(interests)):
        subInterest = interests[interestIndex]
        # ! use JWD to check to see if the compared people's interests are similar
        similarRatio = similar(mainInterest, subInterest)
        if similarRatio > 0.53:
            # if they are similar enough, group them together
            interestSimilarities[mainInterest].append(subInterest)
    # remove interests that have already been compared
    for interest in interestSimilarities[mainInterest]:
        if interest in interests:
            interests.remove(interest)
    checkSimilarInterests(interests, interestSimilarities, counter)


def jobSatisfactionListening(listeningInterestsSimilarities, data):
    total = 0
    satisfaction = {}
    for i in range(0, len(data)):
        jobSatisfaction = data[i][17]
        interest = data[i][42]
        # ! the key entry in this dictionary should be manually changed to a main interest key see the satisfaction rating per interest
        # ! the keys being shown below
        # * podcasts, house music, odesza, maroon 5 what lovers do, tv girl, punk, my anxiety about my career, baseball, spotify, npr, silence
        # * in the report, odesza was categorized as throwback, maroon 5 was categorized as pop, anxiety and silence were combined, baseball was classified as sports
        if interest in listeningInterestsSimilarities['podcasts']:
            total += 0
            if jobSatisfaction not in satisfaction:
                satisfaction[jobSatisfaction] = 1
            elif jobSatisfaction in satisfaction:
                satisfaction[jobSatisfaction] += 1
    for satisfactionKey in satisfaction:
        print(satisfactionKey, ':', satisfaction[satisfactionKey])


def createGraphListening():
    # create graph
    fig, ax = plt.subplots()
    topics = SIMPLIFIED_GROUPS.keys()
    peopleCounts = SIMPLIFIED_GROUPS.values()
    graph = ax.bar(topics, peopleCounts)
    ax.bar_label(graph)
    ax.set_xlabel("Listening Material")
    ax.set_ylabel("People Listening")
    ax.set_title("What People Listen to at Work")
    fig.tight_layout()


def howManyPeoplePerListeningInterest(interests):
    # This will show the main interests that were based off for JWD analysis
    # they were then manually categorized into the groups shown in the report
    total = 0
    totalOverAverage = 0
    for interest in interests:
        total += len(interests[interest])
        if len(interests[interest]) > 197.64:
            # ! uncomment to see the groups that the manually sorted groups are based on using JWD
            # print(interest, ':', len(interests[interest]))
            totalOverAverage += 1
    # * for manual entries, the groups have been simplified into such
    createGraphListening()
    print("Groups over the average amount of people per group", ':', totalOverAverage)
    print('|------------------------------------------------------------------------------------------------------------|')
    print("The manually sorted groups, combining similar groups")
    for manualInterest in SIMPLIFIED_GROUPS:
        print(manualInterest, ':', SIMPLIFIED_GROUPS[manualInterest])
    return


def makeDoubleBarChartSatisfaction():
    # creates a graph showing job satisfaction rates based on listening interests using statistics taken from their interests and satisfaction rates
    # ! the numbers show in satisfaction and disatisfaction using manual math from the output of the function jobSatisfactionListening using the different listening keys
    labels = ['Podcasts', 'House Music', 'Pop Music', 'Throwback']
    satisfaction = [83, 84, 62, 85]
    dissatisfaction = [16, 10, 16, 14]
    x = np.arange((len(labels)))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, satisfaction, width, label='Satisfaction')
    rects2 = ax.bar(x + width/2, dissatisfaction, width, label='Dissatisfaction')
    ax.set_ylabel('Percentage Values')
    ax.set_title('Satisfaction Rates Between Listening Preferences')
    ax.set_xticks(x, labels)
    ax.legend()
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()


def appendTopics(data, topics):
    for i in range(0, len(data)):
        topic = topics[i].split()[-1]
        data[i].append(topic)


def removePodcasts(interests, interestSimilarities):
    interestSimilarities['podcasts'] = []
    for interest in interests:
        if 'podcast' in interest:
            interestSimilarities['podcasts'].append(interest)
    for interest in interestSimilarities['podcasts']:
        if 'podcast' in interest:
            interests.remove(interest)


def removeSmallStrings(interests):
    smallStrings = []
    for interest in interests:
        if len(interest) <= 2:
            smallStrings.append(interest)
    for smallString in smallStrings:
        if smallString in interests:
            interests.remove(smallString)


def similar(a, b):
    return jellyfish.jaro_winkler_similarity(a, b)


def cleanString(inString):
    for specialChar in CUSTOM_SPECIAL_CHARACTERS:
        inString = inString.replace(specialChar, '')
    inString = ' '.join(inString.split())
    return inString


def main():
    plt.rcParams.update({'font.size': 11.5})
    with open('Data.csv', 'r', encoding='utf-8') as inFile, open('Topics.txt', 'r', encoding='utf-8') as inTopics:
        # the column in the data that shows which is the one for listening interests
        listenIndex = 42
        jobSatisfactionIndex = 17
        # read in data
        data = list(csv.reader(inFile))
        # # ! Covers RQ 1
        # Using Jaro–Winkler distance to measure similarity between strings based off people's interests for what they listen to
        # categorize people into groups baed off this non work related interest metric
        listenInterestList = []
        # collect all peoples interests
        for personIndex in range(0, len(data)):
            interest = (data[personIndex][listenIndex]).lower()
            interest = cleanString(interest)
            listenInterestList.append(interest)
        # remove the column labels
        listenInterestList.pop(0)
        listeningInterestsSimilarities = {}
        # remove interests that are not meaningful
        removeSmallStrings(listenInterestList)
        # for collecting interests, remove podcasts and categorize them into their own interest group
        removePodcasts(listenInterestList, listeningInterestsSimilarities)
        #
        counter = 0
        # see which people have similar interests and group them together
        checkSimilarInterests(listenInterestList, listeningInterestsSimilarities, counter)
        # * The insterests of people categorized by those who share a common listening interest
        # * How many groups of people there are that share the same listening interests
        print('|------------------------------------------------------------------------------------------------------------|')
        print("Amount of different listening interest groups", ':', len(listeningInterestsSimilarities))
        print('|------------------------------------------------------------------------------------------------------------|')
        howManyPeoplePerListeningInterest(listeningInterestsSimilarities)
        print('|------------------------------------------------------------------------------------------------------------|')
        print("Job satisfaction for an example group : podcasts")
        jobSatisfactionListening(listeningInterestsSimilarities, data)
        makeDoubleBarChartSatisfaction()
        # ! Cover RQ 2
        print('|------------------------------------------------------------------------------------------------------------|')
        print("Running LDA")
        # remove the column labels
        data.pop(0)
        # convert the data lists into a string format
        baseData = [' '.join(i) for i in data]
        # create vectorizer
        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1))
        # fit transform
        x = vectorizer.fit_transform(baseData)
        # create and fit the LDA model
        model = lda.LDA(n_topics=6, random_state=1)
        model.fit(x)
        doc_topic = model.doc_topic_
        # * This was output to the file output.txt via terminal >> command, I did not have it rewrite files as that is
        # * an expensive operation when running a script multiple times
        # * The following is an example to print out examples for group 5
        # * the hard coded number can be changed to show each topic group sorted from LDA
        for i in range(len(baseData)):
            if doc_topic[i].argmax() == 5:
                # * print which topic the value belongs to, currently commented out as this is an example
                # print(f'Cluster {i}: Topic ', doc_topic[i].argmax())
                # * print out the person who is in the cluster
                print(data[i])
        # show graphs
        plt.show()


main()
