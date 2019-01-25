#!/user/bin/env python

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

#Default settings
maxTweets = 0
epochsTrain = 0
thisTemp = 0.0
userName = ''
generateAtEnd = 8

#Set up list
tweetSet = []

#Figure out if we're training a new model or using one already created
trainerStr = input( "Train a new model or use an existing one?\n1 - Train\n2 - Existing model\n" )
while trainerStr is not "1" and trainerStr is not "2":
	print( "Invalid input." )
	trainerStr = input( "Train a new model or use an existing one?\n1 - Train\n2 - Existing model\n" )

#If we're using the new model
if trainerStr is "1":

	#Delete old files lol
	if os.path.isfile( "twitterman_weights.hdf5" ):
		os.remove( "twitterman_weights.hdf5" )
	if os.path.isfile( "twitterman_vocab.json" ):
		os.remove( "twitterman_vocab.json" )
	if os.path.isfile( "twitterman_config.json" ):
		os.remove( "twitterman_config.json" )

	#User input stuff
	print( "Welcome to Pazda's RNN tweet generator!" )

	#Ask for username
	while True:
		userName = input( "Whose Twitter shall we be abusing? Just enter the username, no @.\n" )
	
		try:
			api.get_user( userName )
		except Exception:
			print( "User not found." )
			continue
		else:
			break

	#Ask for epochs
	while epochsTrain > 60 or epochsTrain <= 0:
		try:
			epochsTrain = int( input( "How many passes should the training make? Must be < 60.\n" ) )
		except ValueError:
			print( "Not an integer." )
			continue
			
	#Ask for tweet search depth
	while maxTweets > 3000 or maxTweets <= 0:
		try:
			maxTweets = int( input( "How far down your timeline should we go? Must be < 3000.\n" ) )
		except ValueError:
			print( "Not an integer." )
			continue
			
	#Ask for result temperature
	while thisTemp > 3.0 or thisTemp <= 0:
		try:
			thisTemp = float( input( "Temperature of results? (higher is wackier) Must be < 3.0.\n" ) )
		except ValueError:
			print( "Not a float." )
			continue
	
	#Use tweepy to find tweets. TODO: optimize this?
	i = 0
	for status in tweepy.Cursor( api.user_timeline,id=userName ).items():
		if i > maxTweets - 1:
			break
		#No retweets, replies, or links please!! (Replies are turned back on. To revert, status.in_reply_to_status_id == None )
		if not hasattr( status, 'retweeted_status' ) and "http" not in status.text:
			#TODO: check if already has punctuation first
			status.text = status.text + "."
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
		max_length = 8,
		max_words = 30000,
		dim_embeddings = 100,
		dropout = 0.0,
		word_level = True )
else:
	#Use existing model
	textgen = textgenrnn( weights_path = 'twitterman_weights.hdf5',
		vocab_path = 'twitterman_vocab.json',
		config_path = 'twitterman_config.json' )
		
	#Ask for result temperature
	while thisTemp > 3.0 or thisTemp <= 0.0:
		try:
			thisTemp = float( input( "Temperature of results? (higher is wackier) Must be < 3.0.\n" ) )
		except ValueError:
			print( "Not a float." )
			continue

#User output.. eventually should send to a webpage fingers crossed
print( "generating extra samples" )
generated = textgen.generate( generateAtEnd, temperature=thisTemp, return_as_list=True )

for x in range( generateAtEnd ):
	print( "\n[------------------Tweet " + str( x ) + "---------------------]\n" + generated[ x ] + "\n[-------------------~~~~~----------------------]\n" )
