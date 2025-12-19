import marimo

__generated_with = "0.18.3"
app = marimo.App(width="medium")


@app.cell
def _():
    from dotenv import load_dotenv, find_dotenv
    from google import genai

    # Loads environment variables from `.env` if it exists
    load_dotenv(find_dotenv(usecwd=True), override=True)

    # The `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable is implicitly read by the client
    client = genai.Client()

    initial_prompt = """
    You will be provided with a program that is written in a simple assembly language. The program will contain 2 blocks of instructions that are commented out and marked with corresponding block numbers 1 and 2. Here is the program:

    ```asm
    MOV R1, 12
    MOV R2, 35
    MOV R3, 7
    MOV R4, 10

    MUL R3, 3
    ADD R1, 5

    label_a:
    INC R1
    SUB R4, 7
    JMP label_b

    label_b:
    ; === BLOCK 1 ===
    ; DEC R3
    ; CMP R3, R1
    ; JL label_c
    ; === BLOCK 1 ===
    CMP R4, R1
    JE label_d
    ADD R4, 13
    JMP label_a

    label_c:
    ADD R4, 7
    DEC R1

    label_d:
    ; === BLOCK 2 ===
    ; CMP R4, R3
    ; JE label_c
    ; SUB R3, 2
    ; DEC R1
    ; === BLOCK 2 ===
    SUB R4, 6
    ADD R2, R4
    SUB R4, 3
    ```

    Question: Which block of instructions should be included in the program so that after the program's execution, the final values stored in the registers R1, R2, R3 and R4 are all divisible by the nearest prime number to 0b10100?

    Using the correct `block_number`, final register states `R1, R2, R3, R4` and the `target_prime_number`, calculate `value` using the expression below:

    `value = sum(block_number, R1, R2, R3, R4, target_prime_number)`

    **You must end your response with the following sentence: "My answer is {value}." (e.g. "My answer is 2.")**
    """

    CORRECT_ANSWER = '135'
    return CORRECT_ANSWER, client, initial_prompt


@app.cell
def _(CORRECT_ANSWER, client, initial_prompt):
    def prompt_weaker_model(prompt):
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )

        # Extract the model's answer from the response
        answer = response.text.rsplit("My answer is", 1)[-1].strip().rstrip(".")

        return (response.text, answer)

    def improve_prompt(prompt, response):
        optimization_prompt = f"""
    A Large Language Model (LLM) was prompted to solve a task but its solution was incorrect.

    #### Prompt
    {prompt}

    #### The Model's Response
    {response}

    #### Correct Solution
    The correct solution to the problem is {CORRECT_ANSWER}.

    #### Your task
    You are tasked with improving the provided prompt so that the model performs better on the given task. Please make sure that the improved prompt still instructs the model to end its response in the correct manner (e.g. "My answer is 2.").
    """

        response_schema = {
            'properties': {
                'improved_prompt': {
                    'description': 'The text of the improved prompt',
                    'title': 'Improved Prompt',
                    'type': 'string',
                },
                'explanation': {
                    'description': "A summary of how the original prompt was improved upon",
                    'title': 'Explanation',
                    'type': 'string',
                },
            },
            'required': ['improved_prompt', 'explanation'],
            'title': 'Improved Prompt Response Object',
            'type': 'object',
        }

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=optimization_prompt,
            config={
                'response_mime_type': 'application/json',
                'response_json_schema': response_schema
            },
        )

        response = response.parsed

        return (response["improved_prompt"], response["explanation"])

    # Function for output section formatting
    def PRINT_BANNER(banner_title):
        leading_spaces = (50 - len(banner_title)) // 2
        print("=" * 50)
        print(" " * leading_spaces, banner_title)
        print("=" * 50)

    def PRINT_RESULT(answer):
        PRINT_BANNER("RESULT")
        if answer == CORRECT_ANSWER:
            print(f"\nThe model answered with the correct solution, which is '{CORRECT_ANSWER}'.\n")
        else:
            print(f"\nThe model's answer was incorrect. Its answer was '{answer}' but the correct solution is '{CORRECT_ANSWER}'.\n")

    # Print the prompt that we are about to send
    PRINT_BANNER("INITIAL PROMPT")
    print(initial_prompt)
    response, answer = prompt_weaker_model(initial_prompt)

    # Print whether the model answered correctly
    PRINT_RESULT(answer)

    # If the model was wrong, try improving the prompt with the help of a stronger model for up to 3 iterations
    if answer != CORRECT_ANSWER:
        prompt = initial_prompt
        for i in range(3):
            print("Improving the prompt...\n")
            prompt, explanation = improve_prompt(prompt, response)
            PRINT_BANNER("IMPROVED PROMPT")
            print(f"\n{prompt}\n")
            PRINT_BANNER("EXPLANATION")
            print(f"\n{explanation}\n")
            response, answer = prompt_weaker_model(prompt)
            PRINT_RESULT(answer)
            if answer == CORRECT_ANSWER:
                break
            
    client.close()
    return


if __name__ == "__main__":
    app.run()
