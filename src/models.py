import openai

from src.utils import sleep


def chat_model(prompt: str, model="gpt-3.5-turbo", temperature: float = 1) -> str:
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message.content
    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        sleep(3)
        return chat_model(prompt)
    except openai.error.Timeout as e:
        print(f"OpenAI API request timed out: {e}")
        sleep(3)
        return chat_model(prompt)
    except openai.error.APIConnectionError as e:
        print(f"OpenAI API request failed to connect: {e}")
        sleep(3)
        return chat_model(prompt)
    except openai.error.RateLimitError as e:
        print(f"OpenAI API request exceeded rate limit: {e}")
        sleep(5)
        return chat_model(prompt)
    except Exception as e:
        if str(e) == "The server is overloaded or not ready yet.":
            print(f"OpenAI API server is overloaded or not ready yet: {e}")
            sleep(10)
            return chat_model(prompt)
        else:
            print(f"Unexpected error: {e}")
