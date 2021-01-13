import logging
import os

import telebot
from flask import Flask, request
from image_extraction.grid_extraction import Color
from main import solve
from PIL import Image

TOKEN = str(os.environ.get("TOKEN", "where-is-my-token"))
bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

# handle commands, /start
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Hello, " + message.from_user.first_name)


# handle all messages, echo response back to users
@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_all_message(message):
    global color
    global target_pos
    try:
        color_string, y, x = message.text.split()
        color = {"r": Color.RED, "b": Color.BLUE, "g": Color.GREEN,
                 "y": Color.YELLOW}[color_string[0].lower()]
        target_pos = int(y), int(x)
        #bot.reply_to(message, f"Search parameters set on bot {color.name}, target position: {target_pos}")

        save_dir = "telegram_files"
        os.makedirs(save_dir, exist_ok=True)
        input_image_path = f"{save_dir}/image.jpg"
        output_image_path = f"{save_dir}/result.jpeg"

        solve_request(message, input_image_path, output_image_path)

    except:
        bot.reply_to(
            message, "Send a picture of the boardgame, or give a bot color and a target \"COLOR y x\" \nEx: \"BLUE 0 0\"")


@bot.message_handler(func=lambda message: True, content_types=['photo'])
def photo(message):

    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    save_dir = "telegram_files"
    os.makedirs(save_dir, exist_ok=True)
    input_image_path = f"{save_dir}/image.jpg"
    output_image_path = f"{save_dir}/result.jpeg"

    # save new image
    with open(input_image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    solve_request(message, input_image_path, output_image_path)


def solve_request(message, input_image_path, output_image_path):

    global color
    global target_pos

    # Load image
    initial_img = Image.open(input_image_path)

    bot.reply_to(
        message, f"Starting the solver\nBot {color.name} tries to reach {target_pos}")

    # Solve challenge
    is_solved = solve(initial_img, color=color,
                      target_pos=target_pos, output_path=output_image_path)

    if is_solved:
        with open(output_image_path, 'rb') as new_file:
            bot.send_photo(chat_id=message.chat.id, photo=new_file)
    else:
        bot.reply_to(message, "No solution found :(")


@app.route('/' + TOKEN, methods=['POST'])
def getMessage():
    app.logger.info("get call on / TOKEN")
    bot.process_new_updates(
        [telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


@app.route("/")
def webhook():
    bot.remove_webhook()
    try:
        app.logger.info("setup webhook")
        bot.set_webhook(url='https://ricochet-bot.herokuapp.com/' + TOKEN)
        return "ok set webhook", 200
    except telebot.apihelper.ApiTelegramException as _:
        return "failed setting webhook", 200


if __name__ == "__main__":

    TESTING = False
    color = Color.YELLOW
    target_pos = (8, 9)

    if TESTING:
        bot.remove_webhook()
        bot.polling()
    else:
        gunicorn_logger = logging.getLogger('gunicorn.error')
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)
        app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
