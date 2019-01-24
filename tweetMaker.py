#!/user/bin/env python
#t
import tweepy
from textgenrnn import textgenrnn
import re
import string
import os

#Twitter API Keys
consumer_key = ""
consumer_secret = ""

access_token_key = ""
access_token_secret = ""

#Twitter API Setup
auth = tweepy.OAuthHandler( consumer_key, consumer_secret )
auth.set_access_token( access_token_key, access_token_secret )
api = tweepy.API( auth )

#Config
trainThisTime = False
maxTweets = 3000
userName = 'EluTranscendent'
generateAtEnd = 8
thisTemp = 0.42
epochsTrain = 30

#Set up lists (context labels are not working, and are unnecessary)
tweetSet = []
#context_labels = []

if trainThisTime:

	#Delete old files lol
	if os.path.isfile( "twitterman_weights.hdf5" ):
		os.remove( "twitterman_weights.hdf5" )
	if os.path.isfile( "twitterman_vocab.json" ):
		os.remove( "twitterman_vocab.json" )
	if os.path.isfile( "twitterman_config.json" ):
		os.remove( "twitterman_config.json" )

	#Use tweepy to find tweets. TODO: optimize this?
	i = 0
	for status in tweepy.Cursor( api.user_timeline,id=userName ).items():
		if i > maxTweets - 1:
			break
		#No retweets, replies, or links please!! (Replies are turned back on. To revert, status.in_reply_to_status_id == None )
		if not hasattr( status, 'retweeted_status' ) and "http" not in status.text:
			tweetSet.append( re.sub( r"(?:\@|https?\://)\S+", "", status.text ) )
			#context_labels.append( userName )
			i += 1

	#debugging
	print( str( len( tweetSet ) ) )

	#Set up RNN model
	textgen = textgenrnn( name='twitterman' )
	textgen.train_new_model(
		tweetSet,
		#context_labels = context_labels,
		num_epochs = epochsTrain,
		gen_epochs = 0,
		train_size = 0.9,
		batch_size = 128,
		rnn_layers = 3,
		rnn_size = 128,
		rnn_bidirectional = True,
		max_length = 14,
		max_words = 30000,
		dim_embeddings = 100,
		dropout = 0.0,
		word_level = True )
else:
	#Use existing model
	textgen = textgenrnn( weights_path = 'twitterman_weights.hdf5',
		vocab_path = 'twitterman_vocab.json',
		config_path = 'twitterman_config.json' )

#User output.. eventually should send to a webpage fingers crossed
print( "generating extra samples" )
generated = textgen.generate( generateAtEnd, temperature=thisTemp, return_as_list=True )

for x in range( generateAtEnd ):
	print( "\n[------------------Tweet " + str( x ) + "---------------------]\n" + generated[ x ] + "\n[-------------------~~~~~----------------------]\n" )
