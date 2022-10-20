from telegram.ext.updater import Updater
from telegram.update import Update
from telegram.ext.callbackcontext import CallbackContext
from telegram.ext.commandhandler import CommandHandler
from telegram.ext.messagehandler import MessageHandler
from telegram.ext.filters import Filters
from handleMsg import Message
from handleMsg import User
import re
import train_classifier

targets_classifiers, name_classifers = train_classifier.load_all_classfiers_in_folder('./Save/')
label_classifier = train_classifier.load_model('./Save/label')

print("Loading bot...")

API_KEY = "5442072729:AAFp2AboU1g6_VSi-yhPfFSagdCABphaA8s"

updater = Updater(API_KEY,
				use_context=True)

user_list = []

def start(update: Update, context: CallbackContext):
	user = update.message.from_user
	print("Start command launched by @"+ user.username)
	update.message.reply_text(
		"Hi sir @"+ user.username + ", welcome to the most moderate telegram bot. You will be surprised by my restraint")

def help(update: Update, context: CallbackContext):
	update.message.reply_text("""Available Commands :-
	/help - To get this help
	/start - To start the bot""")

def unknown(update: Update, context: CallbackContext):
	update.message.reply_text(
		"Sorry '%s' is not a valid command" % update.message.text)


def unknown_text(update: Update, context: CallbackContext):
	user = User(update.message.from_user.id, update.message.from_user.username)
	msg = Message(update.message.text)
	needAdd = True
	for user_ in user_list:
			if(user_.id == update.message.from_user.id):
				needAdd = False
				break
	if(needAdd):
		user_list.append(user)
		print("Added user : " + user.__str__())

	isAttack, response = msg.classifyMessage(label_classifier,targets_classifiers, name_classifers)

	if(isAttack):
		update.message.reply_text("@" + user.__str__() + " " + response)
	#else: 
		#update.message.reply_text("@" + user.__str__() + " Good User")
	


updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('help', help))
updater.dispatcher.add_handler(MessageHandler(
	Filters.command, unknown)) # Filters out unknown commands

# Filters out unknown messages.
updater.dispatcher.add_handler(MessageHandler(Filters.text, unknown_text))

updater.start_polling()