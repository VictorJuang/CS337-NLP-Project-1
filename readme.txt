# Github address
```
https://github.com/VictorJuang/CS337-NLP-Project-1.git
```

# Package install

Please run the following command.
```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader stopwords
```

# How to use

Please run the following command.
```
python gg_api.py
```
After running the command, you can enter the year for tweet data. Please make sure that location of tweet data file (e.g., gg2020.json) should be the same as the location that you run the command.

After enter the year, you can choose the information you are interested in. That is, enter the number 1 ~ 9. 
1. Hosts
2. Awards
3. Nominees
4. Presenters
5. Winner
6. Dress (this should take a lot time on 2015, recommend testing on the other years)
7. Sentiment (this should take a lot time on 2015, recommend testing on the other years)
8. Joke
9. Party

Note that you must run Hosts, Nominees, Presenters and Winner first to get results for Dress and Sentiment.

To exit the program, enter 'x'.

# Other notes

For data in 2013, it may take 10 to 30 seconds to run each function. For data in 2015, it may take minutes to run each function. 

We put the autograder results of 1315 on our end in the img folder. And we also put some additional results in this folder.


# Additional Goals
## Red carpet (Dress)

To find out who was best or worst dressed, we first filter the Tweets that includes the keyword "dress" and a corresponding entity (i.e. actor, actress). After that, for the set of sentences referring to the entity, we calculate the average sentiment value (i.e. polarity and subjectivity).

Using the computed polarity and subjectivity, we map each entity onto a point in the 2D Cartesian coordinates, with polarity and subjectivity being x and y axis respectively.

Finally, we find the point closest to (1, 1) to be the best dressed; that closest to (-1, 1) being the worst dressed; the one having the smallest subjectivity being the most discussed, and the one closest to (-1, 0) being the most controversial.

## Humor (Jokes)

We filter all the Tweets containing the keyword "joke", and calculate its distance from (1, 1) in the polarity-subjectivity coordinates after obtaining the Tweet's sentimental values.

We then sort these Tweets based on the distances, and starting from that closest to (1, 1), we record whether any names are referred, and also accumulate its number of appearance.

Finally, we output the name that has been mentioned the most and its corresponding Tweet that is closest to (1, 1).

## Sentiment

For each entity, we first find Tweets that mentioned it, then we calculate their average sentiment score, and eventually map such score to an adjective.

## Party

For the party search, we use the pattern "at XXXX(any number of words) party" to search the tweets, and calculate the frequence of each party and remove the non-sense party through the stopwords defined by ourselves. Then, we select the most frequent mentioned party and take the tweets mentioned this party and calculate the sentiment score.
