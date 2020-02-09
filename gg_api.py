'''Version 0.35'''

import json
import nltk
import re
import string
from nltk.corpus import stopwords 
import demoji
from nltk.chunk import conlltags2tree, tree2conlltags
import spacy
from pathlib import Path
import itertools
import textdistance
from imdb import IMDb
from difflib import SequenceMatcher
import time
import copy
from textblob import TextBlob
import numpy as np
import random

OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

# define of global variable
OFFICIAL_AWARDS = []
TWEETS = []
CLEAN_DATA = []
TWEET_BY_AWARD_DICT = dict()
ia = IMDb()
GG_RESULT = {}

#  divide_tweets many delete some entries in raw data

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    global GG_RESULT
    if "hosts" in GG_RESULT:
        return GG_RESULT["hosts"]
    load_data(year)
    data_clean()
    # search host
    host_list = ['host', 'hosts', 'hosting', 'hosted']
    host_tweet = []
    for tweet in CLEAN_DATA:
        for test_word in host_list:
            if test_word in tweet:
                host_tweet.append(tweet)
                break
    nlp = spacy.load("en_core_web_sm")
    # get name list
    name_list = []
    for tweet in host_tweet:
        doc = nlp(tweet)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name_list.append(ent.text)
    unique_list = list(set(name_list))
    counter = []
    for item in unique_list:
        counter.append(name_list.count(item))
    sort_counter = sorted(range(len(counter)), key=lambda k: counter[k], reverse=True)  
    
    possible_list = []
    for i in range(5):
        name = unique_list[sort_counter[i]]
        possible_list.append(name)
    reduced_possible_list = []
    for i in range(5):
        test = possible_list[i]
        conflict = 0
        for j in range(5):
            if test in possible_list[j] and i != j:
                conflict = 1
                break
        if test not in reduced_possible_list and conflict == 0 and "golden" not in test and "globes" not in test:
            reduced_possible_list.append(test)
    # search first
    hosts = []
    hosts.append(reduced_possible_list[0])
    host_idx = possible_list.index(hosts[0])
    max_num = counter[sort_counter[host_idx]]
    # search second
    name_pair = []
    for name2 in reduced_possible_list:
        if name2 == hosts[0]:
            continue
        count = 0
        for tweet in host_tweet:
            if  name2 in tweet and hosts[0] in tweet:
                count += 1
        if count > max_num * 0.5:
            name_pair.append([hosts[0], name2, count])
    name_pair.sort(key=lambda x: x[2], reverse=True)
    if len(name_pair) > 0:
        hosts = [name_pair[0][0],name_pair[0][1]]    
    GG_RESULT["hosts"] = hosts
    return hosts

def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    global GG_RESULT
    if "awards" in GG_RESULT:
        return GG_RESULT["awards"]
        
    load_data(year)
    data_clean()
    stop_words = set(stopwords.words('english')) 
    start_pattern_best = re.compile(r'[Bb]est ')
    end_pattern = re.compile(r' goes to')
    end_pattern_is = re.compile(r' is')
    end_pattern_winner = re.compile(r' [Ww]inner')    
    best_split = []
    for i in CLEAN_DATA:
        if len(re.findall(start_pattern_best, i)) > 0:
            split = start_pattern_best.split(i)
            if len(split) > 1:
                best_split.append('best ' + split[1])    
    per_split = []
    for tweet in best_split:
        if len(re.findall(end_pattern, tweet)) > 0:
            split = end_pattern.split(tweet)
            if len(split) > 1:
                per_split.append(split[0])
        elif len(re.findall(end_pattern_is, tweet)) > 0:
            split = end_pattern_is.split(tweet)
            if len(split) > 1:
                per_split.append(split[0])
        elif len(re.findall(end_pattern_winner, tweet)) > 0:
            split = end_pattern_winner.split(tweet)
            if len(split) > 1:
                per_split.append(split[0]) 
    # filtering stage
    len_check = []
    for item in per_split:
        new_item = re.sub(r'[Tt].[Vv].','television',item)
        new_item = re.sub(r'[Tt][Vv]','television',new_item)
        new_item = new_item.lower()      
        if any(str.isdigit(c) for c in new_item):
            continue
        item_token = new_item.split()
        # check stop word
        has_stop_word = 0
        count = 0
        cannot_understand = 0
        for token in item_token:
            if len(token) <=3 and token not in stop_words and token != '-':
                cannot_understand = 1
            if token in stop_words:
                has_stop_word = 1
            else:
                count += 1
        if cannot_understand == 1:
            continue
        if 'golden' not in new_item and 'globe' not in new_item and\
        ' and' not in new_item and ' of' not in  new_item and \
        ' oscar' not in new_item and ' been ' not in  new_item and\
        ' the' not in new_item and ' being' not in new_item and \
        ' disney' not in new_item and ' this' not in new_item and\
        ' while ' not in new_item and 'hotel' not in new_item:
            
            # and ' of' not in new_item  and ' over' not in new_item and ' on' not in new_item and ' and' not in new_item:
            if has_stop_word == 1 and count >= 5:
                len_check.append(new_item)
            elif has_stop_word == 0 and count >= 4:
                len_check.append(new_item)  
    # voting stage                    
    Tmp_answerset = []
    valid_awards = [a[0] for a in nltk.FreqDist(len_check).most_common(50)]
    count = 0
    for i in valid_awards:
        word_list = i.split()
        if len(word_list) > 1:
            #print(i)
            if count < 40:
                Tmp_answerset.append(i)
            count += 1
            
        if count >= 40:
            break
    # remove dup
    New_ans = []
    for tmp_tweet in Tmp_answerset:
        conflict_flag = 0
        for ans_tweet in New_ans:
            check_str_ans = ans_tweet.replace(' in a ', ' ')
            check_str_ans = check_str_ans.replace(',', '')
            check_str_ans = check_str_ans.replace(' - ', ' ')
            check_str_tmp = tmp_tweet.replace(' in a ', ' ')
            check_str_tmp = check_str_tmp.replace(',', '')
            check_str_tmp = check_str_tmp.replace(' - ', ' ')
            #print(check_str_tmp)
            if textdistance.levenshtein(check_str_ans, check_str_tmp) < 4:
                conflict_flag = 1
                break
            if check_str_ans in check_str_tmp or check_str_tmp in check_str_ans:
                conflict_flag = 1
                break            
        if conflict_flag == 0 and len(New_ans) < 26:
            New_ans.append(tmp_tweet)    
    awards = New_ans
    GG_RESULT["awards"] = awards
    return awards

def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''
    # Your code here
    global GG_RESULT
    if "nominees" in GG_RESULT:
        return GG_RESULT["nominees"]
        
    load_data(year)
    data_clean()
    stop_words = set(stopwords.words('english'))
    # reward dict
    award_dict = {}
    nominee = {}
    for award_name in OFFICIAL_AWARDS:
        award_dict[award_name] = {}
        reduced_award_name = award_name.replace("award", "")
        reduced_award_name  = reduced_award_name.replace("made for television" , "television")
        reduced_award_name = reduced_award_name.replace(" - ", " ")
        reduced_award_name = reduced_award_name.replace(",", "")
        reduced_award_name = reduced_award_name.replace(" b. ", " ")
        reduced_award_name = reduced_award_name.replace("performance", "")
        reduced_award_name = reduced_award_name.replace(" by ", " ")
        reduced_award_name = reduced_award_name.replace(" an " , " ")
        reduced_award_name = reduced_award_name.replace(" in a " , " ")
        reduced_award_name = reduced_award_name.replace(" or " , " ")
        reduced_award_name = reduced_award_name.replace(" film" , "")
        reduced_award_name = reduced_award_name.replace(" role " , " ")
        reduced_award_name = reduced_award_name.replace(" series " , " ")
        reduced_award_name = reduced_award_name.replace(" for " , " ")
        reduced_award_name = reduced_award_name.replace("best " , "")
        reduced_award_name = reduced_award_name.replace("mini-series", "series")
        reduced_award_name  = reduced_award_name.replace("feature" , "")
        reduced_award_name  = reduced_award_name.replace("motion picture" , "")
        if "act" in award_name and "television" in award_name:
            reduced_award_name  = reduced_award_name.replace("television" , "")
        award_dict[award_name]['search'] = reduced_award_name.split()
        award_dict[award_name]['prohibit'] = []
        if "motion picture" in award_name and "television" not in award_name:
            award_dict[award_name]['prohibit'].append("television")
        if "act" not in award_name:
            award_dict[award_name]['prohibit'].append("actor")
            award_dict[award_name]['prohibit'].append("actress")
        if "drama" not in award_name:
            award_dict[award_name]['prohibit'].append("drama")
        if "comedy" not in award_name:
            award_dict[award_name]['prohibit'].append("comedy")
        if "musical" not in award_name:
            award_dict[award_name]['prohibit'].append("musical")
        if "act" in award_name and "television" in award_name:
            award_dict[award_name]['prohibit'].append("motion")
            award_dict[award_name]['prohibit'].append("picture")
        nominee[award_name] = []  

    # reward pattern
    # forward
    search_not_win = re.compile(r"(.*)(?:(?:not win))")
    search_deserves_to_win = re.compile(r"(.*)(?:(?:deserves to win))")
    search_deserved_to_win = re.compile(r"(.*)(?:(?:deserved to win))")
    search_is_nominated = re.compile(r"(.*)(?:(?:is nominated))")
    search_was_nominated = re.compile(r"(.*)(?:(?:was nominated))")
    search_missed_out = re.compile(r"(.*)(?:(?:missed out))")
    search_misses_out = re.compile(r"(.*)(?:(?:misses out))")
    search_nominated_for = re.compile(r"(?:(?:nominated for))(.*)")
    search_should_have_won = re.compile(r"(.*)(?:(?:should have won))")
    search_winner_in_my =re.compile(r"(.*)(?:(?:winner in my))")
    search_won_t_win = re.compile(r"(.*)(?:(?:won t win))")
    search_doesn_t_win = re.compile(r"(.*)(?:(?:doesn t win))")
    search_should_win = re.compile(r"(.*)(?:(?:sho?u?ld win))")
    search_better_win = re.compile(r"(.*)(?:(?:better win))")
    search_not_get = re.compile(r"(.*)(?:(?:not get))")
    search_was_robbed = re.compile(r"(.*)(?:(?:was robbed))")
    
    # backward
    search_nominee = re.compile(r"(?:(?:nominee ))(.*)")
    search_preferred = re.compile(r"(?:(?:preferred ))(.*)")
    search_prediction = re.compile(r"(?:(?:prediction))(.*)")
    search_beats = re.compile(r"(?:(?:beats ))(.*)")
    search_should_have_gone_to = re.compile(r"(?:(?:should have gone to ))(.*)")   
    stop_word = ['did', 'the', 'who', 'regrettably', 'when', 'me', 'if', 'wtf', 'wonderful', 'but', 'though', 'believe', 'i think', 'best', 'is', 'won', 'for', 'that','this', 'prediction', 'why', 'supporting', 'actor', 'actress', 'to win', 'original song', 'night', 'evening', 'drama', 'film', 'director', 'it s', 'or', 'had','award','he', 'amp', 'televisions' ]
    stop_splitter = re.compile(r'\bdid\b|\bthe\b|\bwho\b|\bwhen\b|\bregrettably\b|\bme\b|\bif\b|\bwtf\b|\bwonderful\b|\bbut\b|\bthough\b|\bbelieve\b|\bi think\b|\bbest\b|\bis\b|\bwon\b|\bfor\b|\bthat\b|\bthis\b|\bprediction\b|\bwhy\b|\bsupporting\b|\bactor\b|\bactress\b|\bsho?u?ld\b|\bto win\b|\boriginal song\b|\bnight\b|\bevening\b|\bdrama\b|\bfilms?\b|\bdirector\b|\bit s\b|\bor\b|\bhad\b|\baward\b|\bs?he\b|\bamp\b|\bcomedy\b|\bmusical\b|\bout\b|\bwins?\b|\bs\b|\btelevisions?\b|\banimated\b|\bmovies?\b|\bpicture\b|\bnominees?\b|\bforeign language\b|\bmotion\b|\awards?\b|\bwhere\b|\bin a\b|\bwinner\b')    

    def search_pattern(relist, inputstr, idx):
        if len(re.findall(relist, inputstr)) > 0:
            for substr in re.findall(relist, inputstr):
                tmp = re.split(stop_splitter, substr)
                tmp_clean = [x.strip() for x in tmp if x != "" and x != " "]
                #newstr = list(filter("", re.split(stop_splitter, substr)))[-1].strip()
                #print(substr)
                if len(tmp_clean) > 0:
                    newstr = tmp_clean[idx]
                    if len(newstr.split()) <= 3 and newstr not in stop_words and newstr not in string.punctuation:
                        #print(newstr)
                        if "golden" in newstr or "globes" in newstr:
                            continue
                        nominee[award].append(newstr)
    
    nominee = {}
    for award_name in OFFICIAL_AWARDS:
        nominee[award_name] = []
    for award in award_dict:
        #print(award)
        #print(award_dict[award]['search'])
        #print("----------------------")
        if award == 'cecil b. demille award':
            continue
        name_list = []
        for tweet in CLEAN_DATA:
            # search perfect match 
            if all(word in tweet.lower() for word in award_dict[award]['search']) and \
            not any(word in tweet.lower() for word in award_dict[award]['prohibit']) and \
            'oscar' not in tweet.lower() and 'disney' not in tweet.lower():
                search_pattern(search_not_win, tweet.lower(), -1)
                search_pattern(search_deserves_to_win, tweet.lower(), -1)
                search_pattern(search_deserved_to_win, tweet.lower(), -1)
                search_pattern(search_is_nominated, tweet.lower(), -1)
                search_pattern(search_was_nominated, tweet.lower(), -1)
                search_pattern(search_missed_out, tweet.lower(), -1)
                search_pattern(search_misses_out, tweet.lower(), -1)
                search_pattern(search_should_have_won, tweet.lower(), -1)
                search_pattern(search_winner_in_my, tweet.lower(), -1)
                search_pattern(search_won_t_win, tweet.lower(), -1)
                search_pattern(search_better_win, tweet.lower(), -1)
                search_pattern(search_not_get, tweet.lower(), -1)
                search_pattern(search_was_robbed, tweet.lower(), -1)
                
                search_pattern(search_nominee, tweet.lower(), 0)
                search_pattern(search_preferred, tweet.lower(), 0)
                search_pattern(search_prediction, tweet.lower(), 0)
                search_pattern(search_beats, tweet.lower(), 0)
                search_pattern(search_should_have_gone_to, tweet.lower(), 0)   

    nominees = copy.deepcopy(nominee)
    for award in nominees:
        nominees[award] = list(set(nominees[award]))
    GG_RESULT["nominees"] = nominees
    return nominees

def get_winner(year):
    global GG_RESULT
    if "winner" in GG_RESULT:
        return GG_RESULT["winner"]
        
    stop_list_people = ['asian','series','the', 'best', '-', 'award', 'for', 'or', 'made', 'in', 'a', 'by', 'performance', 'an', 'golden', 'globes', 'role', 'motion', 'picture', 'best', 'supporting']
    #stop_list_people =['Motion Picture','Best Actor','Best Supporting']

    divide_tweets(year)
    
    winners = {}
    for award in OFFICIAL_AWARDS:
        winners[award] = []
    

    # print(tweet_award_dict)

    name_pattern = re.compile(r'[A-Z][a-z]+\s[A-Z][a-z]+')
    award_list_person = []
    for award in OFFICIAL_AWARDS:
        for person in ["actor", "actress", "demille", "director"]:
            if person in award:
                award_list_person.append(award)

    for award in award_list_person:
        for tweet in TWEET_BY_AWARD_DICT[award]:
            names = re.findall(name_pattern, tweet)
            for name in names:
                flag = False
                for name_item in name.lower().split():
                    if name_item in stop_list_people:
                        flag = True
                if flag == False:
                    winners[award] = winners[award] + [name]

    freq = {}
    for award in award_list_person:
        freq[award] = nltk.FreqDist(winners[award])

    # winner list for the rest
    award_list_not_person = []
    for award in OFFICIAL_AWARDS:
        if award not in award_list_person:
            award_list_not_person.append(award)

    for award in award_list_not_person:
        
        winner_stoplist = ['globes', 'at', 'and', 'Motion', 'Picture', 'Best', 'Supporting', '-', 'animated', 'best', 'comedy', 'drama', 'feature', 'film', 'foreign', 'globe', 'goes', 'golden', 'motion', 'movie', 'musical', 'or', 'original', 'picture', 'rt', 'series', 'song', 'television', 'to', 'tv', 'movies']
        bigrams_list = []
        ignore_list = ["@", "#"]
        post_process = ['wins', 'goldenglobes']

        for tweet in TWEET_BY_AWARD_DICT[award]:

            tweet = re.sub(r'[^\w\s]', '', tweet)
            if tweet[0:2] == "RT":
                #print (tweet)
                continue

            bigram = nltk.bigrams(tweet.split())

            temp = []
            for item in bigram:
                if item[0].lower() not in winner_stoplist and item[1].lower() not in winner_stoplist:
                    temp.append(item)

            for item in temp:
                if item[0][0] not in ignore_list and item[1][0] not in ignore_list:
                    bigrams_list.append(item)

#         print(bigrams_list)

        freq[award] = nltk.FreqDist([' '.join(item) for item in bigrams_list])

    for award in OFFICIAL_AWARDS:
        #print(freq[award].most_common(1))
        temp_winner = freq[award].most_common(1)[0][0]
        imdb_flag = True
        for word in post_process:
            if word in temp_winner.lower().split():
                #print ('check')
                #print (temp_winner)
                temp_winner = temp_winner.lower().replace(word, '').strip()
                #print (temp_winner)
                #print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                imdb_flag = False
                break
        # winners[award] = temp_winner.lower()
        if award in award_list_person:
            winners[award] = temp_winner.lower()
        else:
            if imdb_flag == False:
                winners[award] = temp_winner.lower()
            else:
                if award != 'best original song - motion picture':
                    movies = ia.search_movie(temp_winner)
                    ss = ''
                    #print (movies)
                    #print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                    for item in movies:
                        try:
                            if item['year'] == int(year)-1:
                                ss = item['title']
                                break
                        except KeyError:
                            continue
                    if ss == '':
                        winners[award] = temp_winner.lower()
                    else:
                        print (ss)
                        print (temp_winner)
                        print (SequenceMatcher(None, ss, temp_winner).ratio())
                        print ('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')
                        if SequenceMatcher(None, ss, temp_winner).ratio() > 0.85:
                            winners[award] = temp_winner.lower()
                        else:
                            winners[award] = ss.lower()
                else:
                    winners[award] = temp_winner.lower()
    GG_RESULT["winner"] = winners                
    return winners

def get_presenters(year):
    #code for presenters
    global GG_RESULT
    if "presenters" in GG_RESULT:
        return GG_RESULT["presenters"]
    start_time = time.time()
    
    divide_tweets(year)
    end_time = time.time()
    print(end_time - start_time)
    print ("Done Preprocessing...")
    
    presenters_dict_by_awards = {}
    stop = ["original", "originals", "best", "screenplay", "feature", "animated", "actress", "motion", "picture", 'lifetime', "series", "comedy", "actor", "supporting", "movie", "foreign", "golden", "award", "goldenglobes", "globes", "film"]

    single_presenter_pattern = re.compile(r'[A-Z][a-z]+\s[A-Z][a-z]+(?=\spresent|\sis\spresenting|\sis\sintroducing|\sgive|\sis\sgiving|\spronounce|\sis\spronouncing|\saward|\sis\sawarding)')
    multiple_presenters_pattern = re.compile(r'[A-Z][a-z]+\s[A-Z][a-z]+\sand\s[A-Z][a-z]*.*\s*[A-Z][a-z]+(?=\spresent|\sintroduc|\sare\sintroducing|\sgive|\sare\sgiving|\spronounce|\sare\spronouncing|\saward|\sare\sawarding)')
        
    for award in TWEET_BY_AWARD_DICT:
        presenters_dict_by_awards[award] = []

        for tweet in TWEET_BY_AWARD_DICT[award]:
            tweet = re.sub(r'&amp;', 'and', tweet)
            tweet = re.sub(r'\s+@\w+', '', tweet)
            multiple_presenters = re.findall(multiple_presenters_pattern, tweet)

            for presenter in multiple_presenters:
                pp = presenter.split(' and ')
                p1 = pp[0]
                if any(word in p1.lower() for word in stop):
                    continue

                pp = presenter.split(' and ')
                pt = pp[1]
                ptt = pt.split(' ')
                pttname = ptt[0:2]
                p2 = ' '.join(pttname)
                if any(word in p2.lower() for word in stop):
                    continue

                person = ia.search_person(p1)
                if person:
                    p1 = person[0]['name'].lower()
                person = ia.search_person(p2)
                if person:
                    p2 = person[0]['name'].lower()
                if p1 not in presenters_dict_by_awards[award]:
                    presenters_dict_by_awards[award].append(p1)
                if p2 not in presenters_dict_by_awards[award]:
                    presenters_dict_by_awards[award].append(p2)

            single_presenter = re.findall(single_presenter_pattern, tweet)
            for presenter in single_presenter:
                if any(word in presenter.lower() for word in stop):
                    continue
                person = ia.search_person(presenter)
                if person:
                    presenter = person[0]['name'].lower()
                if presenter not in presenters_dict_by_awards[award]:
                    presenters_dict_by_awards[award].append(presenter)
    GG_RESULT["presenters"] = presenters_dict_by_awards
    return presenters_dict_by_awards

def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    print("Pre-ceremony processing complete.")
    return
    
    
def load_data(year):
    global OFFICIAL_AWARDS
    global TWEETS
    if not OFFICIAL_AWARDS or not TWEETS:
        print("get")
        get_tweets(year)
        if year == "2013" or year == "2015": #or year == "2020":
            OFFICIAL_AWARDS = OFFICIAL_AWARDS_1315
        else: 
            OFFICIAL_AWARDS = OFFICIAL_AWARDS_1819
    else:
        print("pass load")

def get_tweets(year):
    global TWEETS
    try:
        if year == "2013" or year == "2015"or year == "2018"or year == "2019":
            f = open('gg'+year+'.json')
            data = json.load(f)
            TWEETS = [tweet['text'] for tweet in data]
            #return TWEETS
        else:                                              
            with open('gg'+year+'.json', encoding='utf8') as json_file:
                data = [json.loads(line) for line in json_file]
            TWEETS = [tweet['text'] for tweet in data]
            #return TWEETS
        TWEETS.sort()
        TWEETS = list(TWEETS for TWEETS,_ in itertools.groupby(TWEETS))
    except:
        return False

def strip_all_entities(text):
    new_str = string.punctuation
    new_str = new_str.replace('-','')
    new_str = new_str.replace(',','')
    new_str += '“'
    entity_prefixes = ['@','#']
    for separator in new_str:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)
    
def data_clean():  
    global TWEETS
    global CLEAN_DATA
    if not CLEAN_DATA:
        for tweet_text in TWEETS:
            #tweet_text = tweet['text']
            no_http = re.sub('http://\S+|https://\S+', '', tweet_text)
            remove_tag = strip_all_entities(no_http)
            new_item = re.sub(r'–','-',remove_tag)
            new_item = re.sub(r'[Tt].[Vv].','television',new_item)
            new_item = re.sub(r'[Tt][Vv]','television',new_item) 
            new_item = re.sub(r'RT ','',new_item) 
            #new_item = new_item.translate(str.maketrans('', '', string.punctuation))
            CLEAN_DATA.append(new_item.lower())
        #CLEAN_DATA.sort()
        #CLEAN_DATA = list(set(CLEAN_DATA))
    else:
        print("pass clean")

def divide_tweets(year):

    print("\nProcessing data of year {}".format(year))
    #print("Might take under 4 minutes for 2013 and 5-6 minutes for 2015 ")
    load_data(year)
    data_clean()
    #global TWEETS

    global TWEET_BY_AWARD_DICT
    if not not TWEET_BY_AWARD_DICT:
        return
        
        
    for award in OFFICIAL_AWARDS:
        TWEET_BY_AWARD_DICT[award] = []

    to_delete = ['-', 'a', 'an', 'award', 'best', 'by', 'for', 'in', 'made', 'or', 'performance', 'role',
                 'feature', 'language']

    fresh_names = dict()
    for award in OFFICIAL_AWARDS:
        fresh_names[award] = [[item for item in award.split() if not item in to_delete]]

    for award in OFFICIAL_AWARDS:
        if "television" in award:
            for word in ['tv', 't.v.']:
                extra = award.replace("television", word)
                fresh_names[award].append([item for item in extra.split() if not item in to_delete])

        if "motion picture" in award:
            for word in ['movie', 'film']:
                extra = award.replace("motion picture", word)
                fresh_names[award].append([item for item in extra.split() if not item in to_delete])


        if "film" in award:
            for word in ['motion picture', 'movie']:
                extra = award.replace("film", word)
                fresh_names[award].append([item for item in extra.split() if not item in to_delete])

        if "comedy or musical" in award:
            for word in ['comedy', 'musical']:
                extra = award.replace("comedy or musical", word)
                fresh_names[award].append([item for item in extra.split() if not item in to_delete])


        if "series, mini-series or motion picture made for television" in award:
            for word in ['series', 'mini-series', 'miniseries', 'tv', 'television', 'tv movie', 'tv series', 'television series']:
                extra = award.replace("series, mini-series or motion picture made for television", word)
                fresh_names[award].append([item for item in extra.split() if not item in to_delete])

            

        if "mini-series or motion picture made for television" in award:
            for word in ['television movie', 'mini-series', 'miniseries', 'tv movie']:
                extra = award.replace("mini-series or motion picture made for television", word)
                fresh_names[award].append([item for item in extra.split() if not item in to_delete])

        if "television series" in award:
            for word in ['tv', 't.v.', 'television', 'series']:
                extra = award.replace("television series", word)
                fresh_names[award].append([item for item in extra.split() if not item in to_delete])


        if "television series - comedy or musical" in award:

            for word in ["tv comedy", "tv musical", "comedy series", "t.v. comedy", "t.v. musical", "television comedy", "television musical"]:
                extra = award.replace("television series - comedy or musical", word)
                fresh_names[award].append([item for item in extra.split() if not item in to_delete])

        if "television series - drama" in award:
            for word in ["tv drama", "drama series", "television drama", "t.v. drama"]:
                extra = award.replace("television series - drama", word)
                fresh_names[award].append([item for item in extra.split() if not item in to_delete])

    OFFICIAL_AWARDS.sort(key=lambda s: len(s), reverse=True)
#     print('####################FRESH NAMES##########################')
#     print(fresh_names)
#     print('#########################################################')
    #all(word in tweet.lower() for word in award_dict[award]['search'])
    #for award in OFFICIAL_AWARDS:
    #    tweet_length = len(TWEETS)
    #    for i in range(tweet_length - 1, -1, -1):
    #        tweet = TWEETS[i]
    #        for extra in fresh_names[award]:
    #            flag = True
    #            for word in extra:
    #                if flag == True:
    #                    flag = flag and word.lower() in tweet.lower()
    #
    #            if flag == True:
    #                TWEET_BY_AWARD_DICT[award].append(tweet)
    #                del TWEETS[i]
    #                break
    for award in OFFICIAL_AWARDS:
        tweet_length = len(TWEETS)
        for i in range(tweet_length - 1, -1, -1):
            tweet = TWEETS[i]
            
            for extra in fresh_names[award]:

                if all(word in tweet.lower() for word in extra):
                #extra_tmp = set(extra)
                #tweet_tmp = set(tweet.split())
                #if len(extra_tmp) == len(extra_tmp.intersection(tweet_tmp)):
                    TWEET_BY_AWARD_DICT[award].append(tweet)
                    del TWEETS[i]
                    break

def get_additional(year, addition_str):
    load_data(year)
    data_clean()
    person_list = []
    all_list = []
    if addition_str == "joke":
        additional(person_list, addition_str, load_data_party(year))
        return
    if "hosts" not in GG_RESULT:
        print("please run hosts function first!")
        return
    elif "nominees" not in GG_RESULT:
        print("please run nominees function first!")
        return
    elif "winner" not in GG_RESULT:
        print("please run winner function first!")
        return
    elif "presenters" not in GG_RESULT:
        print("please run presenters function first!")
        return        
        
    for title in GG_RESULT:
        if title == "hosts":
            for per in GG_RESULT[title]:
                person_list.append(per)
                all_list.append(per)
        elif title in [ "nominees", "winner", "presenters"]:
            for award in GG_RESULT[title]:
                if "act" in award or "demille" in award or "director" in award:
                    if title == "winner":
                        person_list.append(GG_RESULT[title][award])
                    else:
                        for per in GG_RESULT[title][award]:
                            person_list.append(per) 
                if title == "winner":
                    all_list.append(GG_RESULT[title][award])
                else:
                    for per in GG_RESULT[title][award]:
                        all_list.append(per) 

                    
    print(person_list)
    if addition_str == "sentiment":
        additional(all_list, addition_str, load_data_party(year))
    else:
        additional(person_list, addition_str, load_data_party(year))

def sentiment(sentences: list) -> tuple:
    '''Can be used on entity's sentiment and red carpet, Humor (find the word dress/joke first)
       best dressed (++), worst dressed (-+), most discussed(.-), most controversial(--),'''
    polarity, subjectivity = 0, 0
    for sentence in sentences:
        tb = TextBlob(sentence.replace("\n", ""))
        polarity += tb.sentiment.polarity
        subjectivity += tb.sentiment.subjectivity
    
    polarity, subjectivity = polarity/len(sentences), subjectivity/len(sentences)

    return polarity, subjectivity

def find_keyword(sentences: list, keyword: str) -> list:
    '''keyword should be in lower case'''
    data = []
    for sentence in sentences:
        tb = TextBlob(sentence)
        if keyword in tb.lower(): data.append(sentence)
    
    return data

def additional(entities: list, typ: str, data: list):
    if typ not in ["dress", "joke", "sentiment", "parties", "act"]: 
        raise Exception

    P, S = [], []

    if typ == "dress" or typ == "sentiment":
        for entity in entities:
            _data = find_keyword(data, entity)

            if typ == "sentiment" and len(_data):
                p, s = sentiment(_data)
                P.append(p)
                S.append(s)
            elif typ == "sentiment" and not len(_data):
                P.append(0)
                S.append(0.5)

            else: 
                __data = find_keyword(_data, typ)
            
                if not len(__data):
                    P.append(0)
                    S.append(0.5)
                else:
                    p, s = sentiment(__data)
                    P.append(p)
                    S.append(s)

    elif typ == "joke":
        _data = find_keyword(data, typ)
        for _d in _data:
            p, s = sentiment([_d])
            P.append(p)
            S.append(s)

    P, S = np.array(P), np.array(S)
    
    if typ == "dress":
        print("best dressed", entities[np.argmin(np.power(P-1, 2) + np.power(S-1, 2))])
        print("worst dressed", entities[np.argmin(np.power(P+1, 2) + np.power(S-1, 2))])
        print("most discussed", entities[np.argmin(S)])
        print("most controversial", entities[np.argmin(np.power(P+1, 2) + np.power(S, 2))])

    elif typ == "joke":
        N = {}
        name_pattern = re.compile(r'[A-Z][a-z]+\s[A-Z][a-z]+')
        _sorted = np.argsort(np.power(P-1, 2) + np.power(S-1, 2))
        for i in _sorted:
            names = re.findall(name_pattern, _data[i])
            if len(names): 
                for n in names:
                    if n in N: 
                        N[n]["freq"] += 1
                        N[n]["tweets"].append(i)
                    else: 
                        N[n] = {}
                        N[n]["freq"] = 1
                        N[n]["tweets"] = [i]
        
        f, name, tweet = 0, "", ""
        tweet_idx = 0
        for _name in N:
            if int(N[_name]["freq"]) > f and _name != "Golden Globes":
                f = int(N[_name]["freq"])
                name = _name
                tweet_idx = N[_name]["tweets"][0]

        print(name, _data[tweet_idx])

    elif typ == "sentiment":
        for idx, entity in enumerate(entities):
            if P[idx] < 0:
                if S[idx] < 0.25:
                    word = random.choice([
                        "controversial",
                        "weird"
                    ])
                else:
                    word = random.choice([
                        "unpleasant",
                        "irritating"
                    ])
            elif P[idx] < 0.3:
                if S[idx] < 0.515:
                    word = random.choice([
                        "neutral",
                        "acceptable",
                        "appreciated",
                    ])
                else:
                    word = random.choice([
                        "fair",
                        "agreeable"
                    ])
            else:
                if S[idx] < 0.5:
                    word = "impressive"
                else:
                    word = "delightful"

            print(entity, word)

def load_data_party(year):
    if year == "2013" or year == "2015"or year == "2018"or year == "2019":
        f = open('gg'+year+'.json')
        data = json.load(f)
        TWEETS = [tweet['text'] for tweet in data]
        #return TWEETS
    else:                                              
        with open('gg'+year+'.json', encoding='utf8') as json_file:
            data = [json.loads(line) for line in json_file]
        TWEETS = [tweet['text'] for tweet in data]
            #return TWEETS
    TWEETS.sort()
    TWEETS = list(TWEETS for TWEETS,_ in itertools.groupby(TWEETS))
    return TWEETS

def find_party(tweet):
    ss = dict()
    party_pattern = re.compile(r'at [a-zA-Z\s]+(?=\sparty)')
    for item in tweet:
        pp = re.findall(party_pattern, item)
        for party in pp:
            if party[3:] not in ss:
                ss[party[3:]] = 1
            else:
                ss[party[3:]] += 1
    
    ordered_p_tweet = sorted(ss.items(), key=lambda x: x[1], reverse=True)
    stop_party = ['after', 'window']
    clean_p_tweet = []
    for item in ordered_p_tweet:
        flag = True
        for word in stop_party:
            if word in item[0].lower().split():
                flag = False
                break
        if flag:
            clean_p_tweet.append(item)
                
    nonsense = ['a', 'the', 'an', 'his', 'her', 'their', 'my', 'a golden globe', 'the golden globes','our golden globes', 'golden globe', 'golden globes', 'your golden globes']
    _clean_p_tweet = []
    for item in clean_p_tweet:
        flag = True
        for w in nonsense:
            if w == item[0].lower():
                flag = False
                break
        if flag:
            _clean_p_tweet.append(item)
        
    
    return _clean_p_tweet

def find_party_main_function(year):
    data = load_data_party(year)
    #data = clean_data(data)
    p_tweet = find_party(data)
    #print (len(p_tweet))

    most_party_twitter = []
    for item in data:
        if p_tweet[0][0] in item:
            most_party_twitter.append(item)
    print ("Party People Most Talk:", p_tweet[0][0])
    print ("Sentiment Score for the Party:", sentiment(most_party_twitter))
    
    
def main():
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    # Your code here
    pre_ceremony()
    select_year = ""
    while not select_year.isdigit():
        print("Enter an year. Or enter character \"x\" to leave.")
        select_year = input()
        if select_year == "x":
            print("Exit the program.")
            return
        if not select_year.isdigit():
            print("Wrong input! Please enter a number or enter character \"x\"")
        if not Path('gg' + str(select_year) + '.json').is_file():
            print("File not found! Please enter a valid year!")
            select_year = ""     
    print('+---------------------+')
    print('| Selected year: ' + str(select_year))
    print('+---------------------+')
    
    while True:
        print('Enter a number below to get corresponding information.')
        print('1. Hosts')
        print('2. Awards')
        print('3. Nominees')
        print('4. Presenters')
        print('5. Winner')
        print('6. Dress')
        print('7. Sentiment')
        print('8. Joke')
        print('9. Party')
        option = input()
        if option == "x":
            break
        elif option == "1":
            print(get_hosts(select_year))
        elif option == "2":
            print(get_awards(select_year))
        elif option == "3":
            print(get_nominees(select_year))
        elif option == "4":
            print(get_presenters(select_year))
        elif option == "5":
            print(get_winner(select_year))
        elif option == "6":
            get_additional(select_year, "dress")
        elif option == "7":
            get_additional(select_year, "sentiment")
        elif option == "8":
            get_additional(select_year, "joke")
        elif option == "9":
            find_party_main_function(select_year)               
        else:
            print('Wrong input! Program terminated.')
           
    
    return

if __name__ == '__main__':
    main()
