from typing import List, Dict, Any

def get_external_llm_messages(
    base_url_host: str, 
    additional_instructions: str, 
    transcript: List[Dict[str, Any]], 
    agent_identity: str
) -> List[Dict[str, Any]]:
    is_openai = "openai.com" in base_url_host
    additional_instructions = f"\n\n## Instructions:\n{additional_instructions}" if additional_instructions else ""
    messages = [
        {
            "role": "developer" if is_openai else "system",
            "content": f"""You are a friendly assistant engaging in a spoken telephone conversation with a user.

## Response Format:
- Respond with only one sentence at a time.
- If you think the user still has more to say before you respond, respond with a simple backchannel (e.g. yeah, mhm) or with [silence].
- Likewise, if the user responds with a backchannel (e.g. yeah, mhm) or with [silence], you may continue your previous response.{additional_instructions}"""
        }
    ]
    for turn in transcript:
        if turn["speaker"] != agent_identity:
            if messages[-1]["role"] == "user":
                messages.append({
                    "role": "assistant",
                    "content": "[silence]"
                })                
            messages.append({
                "role": "user",
                "content": turn["text"]
            })
        else:
            if messages[-1]["role"] != "user":
                messages.append({
                    "role": "user",
                    "content": "[silence]"
                })
            messages.append({
                "role": "assistant",
                "content": turn["text"]
            })
    # Ensure the last message is from the user
    # This is important for the chat-templated LLM to respond correctly
    if messages[-1]["role"] != "user":
        messages.append({
            "role": "user",
            "content": "[silence]"
        })
    return messages