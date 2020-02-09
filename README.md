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