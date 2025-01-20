import torch
from transformers import pipeline
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters

# Load TinyLlama text-generation pipeline
pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Define the bot's response logic
async def handle_message(update: Update, context):
    user_message = update.message.text  # Get the user's message

    # Debugging: Print the received message
    print(f"Received message: {user_message}")

    # Define the chat template messages
    messages = [
        {
            "role": "system",
            "content": "You are an chatbot created for one homework assignment. You can ask me anything!",
        },
        {"role": "user", "content": user_message},
    ]

    # Format the prompt using the tokenizer's chat template
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Generate the response
    print("Generating response...")  # Debugging: Confirm the bot is processing the response
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    # Extract the generated text
    response = outputs[0]["generated_text"]
    print(f"Generated response: {response}")  # Debugging: Print the generated response

    # Send the response back to the user
    await update.message.reply_text(response)

# Set up the Telegram bot
app = ApplicationBuilder().token("7052606848:AAH2BA5_S5R-RK4xyDjGfNTXAPfNuPzTbjQ").build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

# Run the bot
print("Bot is running...")  # Debugging: Confirm the bot is running
app.run_polling()
