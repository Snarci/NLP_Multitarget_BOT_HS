
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from utility_functions import classify_text
from utility_functions import load_all_classfiers_in_folder
import train_classifier_utility as tcu
# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

print("Loading classifier...")
targets_classifiers=load_all_classfiers_in_folder("trained_classifiers")
label_classifier =tcu.load_classifier("./trained_classifiers/trained_classifier_Labels.sav")
print("loaded classifiers")

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text(
        "Hi there! I'm a bot that can detect if a message contains hate speech or offensive language!"
    )


def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text(
        "This is a bot that can detect if a message contains hate speech or offensive language!"
    )


def echo(update, context):
    """Echo the user message."""
    update.message.reply_text(update.message.text)


def classify_message(update, context):
    user_txt = update.message.text
    update.message.reply_text(classify_text(user_txt, label_classifier, targets_classifiers,0.4, verbose=True))
    




def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater("5467861148:AAGG0yQ8grtcAlxlX40F7qW-eeujP-klOmI", use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, classify_message))
    #dp.add_handler(MessageHandler(Filters.regex(r"^[@][\w]+"), user_hate_speech))
    #dp.add_handler(MessageHandler(Filters.regex(r"^[#][\w]+"), topic_hate_speech))
    #dp.add_handler(MessageHandler(Filters.text, incorrect_message))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == "__main__":
    main()