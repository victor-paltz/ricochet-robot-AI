import logging
import os
from typing import Tuple

import telebot
from flask import Flask, request
from image_extraction.Color import Color
from main import solve
from PIL import Image

TOKEN = str(os.environ.get("TOKEN", "where-is-my-token"))
bot = telebot.TeleBot(TOKEN)
app = Flask(__name__)

SAVE_DIR = "telegram_files"
os.makedirs(SAVE_DIR, exist_ok=True)

HELP_MESSAGE = ("Welcome to the ricochet-robot solver!\n\n"

                "1) Send a picture of the boardgame.\n"
                "\t-> Take the picture from above\n"
                "\t-> Avoid to hide the walls with the robots\n"
                "\t-> Try to use an uniform light\n"
                "\t-> Avoid a background with lines\n\n"

                "2) Send a message of the type: \"COLOR y x\" to get the shortest solution to move the COLOR robot to the case in column number y and line number x\n\n"

                "3) Send a new request on the current picture with step 1) or send a new boardgame picture with step 1)\n\n\n"

                "Ex: If you want to know how the BLUE robot can go to the TOP RIGHT corner, send \"BLUE 1 16\" (1 -> first line, 16 -> last column)\n\n"
                "Pro Tips, \"YELLOW 12 5\", \"Y 12 5\", \"y 12 5\", \"y125\" are all accepted")

# handle commands, /start
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(
        message, f"Hello {message.from_user.first_name},\n" + HELP_MESSAGE)


# handle all messages, echo response back to users
@bot.message_handler(func=lambda message: True, content_types=['text'])
def handle_all_message(message):
    global color
    global target_pos

    color_string_dict = {"r": Color.RED, "b": Color.BLUE, "g": Color.GREEN,
                         "y": Color.YELLOW}

    try:
        color_string, y, x = match_query(message.text)
    except:
        bot.reply_to(
            message, "Sorry, I don't understand your query, send /start to get some help")
        return

    if color_string not in color_string_dict:
        bot.reply_to(
            message, "Wrong color, color available are RED, BLUE, GREEN and YELLOW")
        return

    if not (1 <= y <= 16) or not (1 <= x <= 16):
        bot.reply_to(
            message, "Wrong target position, lines and colums are in range 1 -> 16")
        return

    if (7 <= y <= 8) and (7 <= x <= 8):
        bot.reply_to(
            message, "Wrong target position, cannot reach cases in the center")
        return

    color = color_string_dict[color_string]
    target_pos = int(y-1), int(x-1)

    input_image_path = f"{SAVE_DIR}/{message.chat.id}-input.jpg"
    output_image_path = f"{SAVE_DIR}/{message.chat.id}-output.jpg"

    if os.path.isfile(input_image_path):
        solve_request(message, input_image_path, output_image_path)
        return
    else:
        bot.reply_to(
            message, "There is not previously sent picture of the board, please send a picture")
        return


@bot.message_handler(func=lambda message: True, content_types=['photo'])
def photo(message):

    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    input_image_path = f"{SAVE_DIR}/{message.chat.id}-input.jpg"

    # save new image
    with open(input_image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    bot.reply_to(
        message, "New board game updated! you can now type queries such as \"GREEN 16 16\" or \"g1616\"")


def solve_request(message, input_image_path, output_image_path):

    global color
    global target_pos

    # Load image
    initial_img = Image.open(input_image_path)

    bot.reply_to(
        message, f"Starting the solver\n{color.name} bot tries to reach case {target_pos[0]+1}, {target_pos[1]+1}")

    # Solve challenge
    is_solved = solve(initial_img, color=color,
                      target_pos=target_pos, output_path=output_image_path)

    if is_solved:
        with open(output_image_path, 'rb') as new_file:
            bot.send_photo(chat_id=message.chat.id, photo=new_file)
    else:
        bot.reply_to(
            message, "No solution found :(\n(Max depth exploration: 10)")


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


def match_query(text: str) -> Tuple[str, int, int]:

    text = text.replace(",", " ").replace("-", " ").strip()

    color = ""
    for i, c in enumerate(text):
        if not c.isalpha():
            break
        color += c

    if not color:
        raise ValueError()

    color = color[0].lower()

    rest_text = [x for x in text[i:].split(" ") if x]

    if len(rest_text) == 2:
        a = "".join(y for y in rest_text[0] if y.isnumeric())
        b = "".join(y for y in rest_text[1] if y.isnumeric())
        if not a or not b:
            raise ValueError()
        return color, int(a), int(b)
    elif len(rest_text) == 1:
        ab = rest_text[0]
        if len(ab) == 4:
            return color, int(ab[:2]), int(ab[2:])
        if len(ab) == 2:
            return color, int(ab[0]), int(ab[1])
        if len(ab) == 3:
            a1, b1 = int(ab[:2]), int(ab[2])
            a2, b2 = int(ab[0]), int(ab[1:])

            if 1 <= a1 <= 16 and 1 <= b1 <= 16:
                if 1 <= a2 <= 16 and 1 <= b2 <= 16:
                    raise ValueError()
                else:
                    return color, int(a1), int(b1)
            elif 1 <= a2 <= 16 and 1 <= b2 <= 16:
                return color, int(a2), int(b2)

        else:
            raise ValueError()
    else:
        raise ValueError()


if __name__ == "__main__":

    TESTING = False
    color = Color.YELLOW
    target_pos = (8, 9)

    if TESTING:
        bot.remove_webhook()
        bot.polling()
        bot.set_webhook(url='https://ricochet-bot.herokuapp.com/' + TOKEN)
        print("done")
    else:
        gunicorn_logger = logging.getLogger('gunicorn.error')
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)
        app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))
