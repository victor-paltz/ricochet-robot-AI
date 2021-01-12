import os

from flask import Flask, request
import telebot
from main import solve
from PIL import Image
from image_extraction.grid_extraction import Color

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
    bot.reply_to(message, "Send a picture of the boardgame")


@bot.message_handler(func=lambda message: True, content_types=['photo'])
def photo(message):

    save_dir = "telegram_files"
    os.makedirs(save_dir, exist_ok=True)
    input_image_path = f"{save_dir}/image.jpg"
    output_image_path = f"{save_dir}/result.jpeg"

    file_id = message.photo[-1].file_id
    file_info = bot.get_file(file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    with open(input_image_path, 'wb') as new_file:
        new_file.write(downloaded_file)

    # Load image
    initial_img = Image.open(input_image_path)

    bot.reply_to(message, "Starting the solver")

    # Solve challenge
    is_solved = solve(initial_img, color=Color.YELLOW,
                      target_pos=(8, 9), output_path=output_image_path)

    if is_solved:
        with open(output_image_path, 'rb') as new_file:
            bot.send_photo(chat_id=message.chat.id, photo=new_file)
    else:
        bot.reply_to(message, "No solution found :(")


@app.route('/' + TOKEN, methods=['POST'])
def getMessage():
    bot.process_new_updates(
        [telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "transfert ok", 200


@app.route("/")
def webhook():
    bot.remove_webhook()
    try:
        bot.set_webhook(url='https://ricochet-bot.herokuapp.com/' + TOKEN)
        return "ok set webhook", 200
    except telebot.apihelper.ApiTelegramException as _:
        return "failed setting webhook", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))

# bot.polling()
