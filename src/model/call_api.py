def call_llm_for_data_cleaning_or_analysis(client, model, prompt):
    response = client.chat_with_prompt_return_text(
        model=model,
        prompt=prompt,
        temperature=1.0,
        response_format={'type': 'json_object'},
    )

    return response