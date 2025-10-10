from typing import Optional, Tuple, Dict, List
from transformers import PreTrainedTokenizerFast
from .utils.llamacpp_utils import LlamaForAlternatingCodeChannels

class ExternalLLMHandler:
    def __init__(
        self, 
        llm: LlamaForAlternatingCodeChannels, 
        tokenizer: PreTrainedTokenizerFast, 
        top_p: float = 0.9,
        seed: Optional[int] = 42,
        additional_instructions: Optional[str] = None,
        agent_opening_text: Optional[str] = None,
    ):
        self.llm = llm
        self.tokenizer = tokenizer
        self.top_p = top_p
        self.seed = seed
        self.agent_opening_text = agent_opening_text

        additional_instructions = f"\n\n## Instructions:\n{additional_instructions}" if additional_instructions else ""
        self.system_message = {
            "role": "system",
            "content": f"""You are a friendly assistant engaging in a spoken telephone conversation with a user.

## Response Format:
- Respond naturally, including backchannels (e.g. yeah, sure, mhm), fillers (e.g. uh, um, hmm) and laughter (e.g. [laughing], [laughs]).
- You can also choose to say nothing, in which case respond with [silence].
- If the user responds with a backchannel (e.g. yeah, sure, mhm) or with [silence], you may continue your previous response.{additional_instructions}"""
        }

        self.dummy_sys_msg, self.dummy_sys_msg_length, self.gen_prompt_input_ids, self.post_eos_input_ids = self._get_chat_template_stuff()
        self.reset()

    def _get_chat_template_stuff(self) -> Tuple[Dict[str, str], int, List[int], List[int]]:
        dummy_messages = [
            {"role": "system", "content": "foo"},
            {"role": "user", "content": "bar"},
        ]
        sys_msg_input_ids = self.tokenizer.apply_chat_template(dummy_messages[:1])
        input_ids = self.tokenizer.apply_chat_template(dummy_messages)
        input_ids_with_gen_prompt = self.tokenizer.apply_chat_template(dummy_messages, add_generation_prompt=True)
        dummy_sys_msg = dummy_messages[0]
        dummy_sys_msg_length = len(sys_msg_input_ids)
        gen_prompt_input_ids = input_ids_with_gen_prompt[len(input_ids):]

        dummy_messages.append({"role": "assistant", "content": "baz"})
        prev_input_ids_length = len(input_ids)
        input_ids = self.tokenizer.apply_chat_template(dummy_messages)
        last_eos_pos = input_ids.index(self.tokenizer.eos_token_id, prev_input_ids_length)
        post_eos_input_ids = input_ids[last_eos_pos+1:]

        return dummy_sys_msg, dummy_sys_msg_length, gen_prompt_input_ids, post_eos_input_ids

    def reset(self) -> None:
        self.llm.init_sampler_for_generate(
            top_p=self.top_p,
            min_p=0.0,
            temp=1.0,
            seed=self.seed,
        )
        self.llm.reset()
        self.messages = [self.system_message]
        if self.agent_opening_text:
            self.messages.extend([
                {
                    "role": "user",
                    "content": "[silence]",
                },
                {
                    "role": "assistant",
                    "content": self.agent_opening_text,
                },
            ],
        )
        self.input_ids = self.tokenizer.apply_chat_template(self.messages)
        self.llm.eval(self.input_ids)

    def process_user_text(self, user_text: str) -> None:
        new_messages = []
        if self.messages[-1]["role"] == "user":
            new_messages.append({
                "role": "assistant",
                "content": "[silence]",
            })                
        new_messages.append({
            "role": "user",
            "content": user_text,
        })
        self.messages.extend(new_messages)
        new_input_ids = self.tokenizer.apply_chat_template([self.dummy_sys_msg] + new_messages)[self.dummy_sys_msg_length:]
        self.input_ids.extend(new_input_ids)
        self.llm.eval(new_input_ids)

    def generate_response(self) -> str:
        if self.messages[-1]["role"] != "user":
            self.messages.append({
                "role": "user",
                "content": "[silence]",
            })
            new_input_ids = self.tokenizer.apply_chat_template([self.dummy_sys_msg] + self.messages[-1:])[self.dummy_sys_msg_length:]
            self.input_ids.extend(new_input_ids)
            self.llm.eval(new_input_ids)
        # Generate
        self.input_ids.extend(self.gen_prompt_input_ids)
        self.llm.eval(self.gen_prompt_input_ids[:-1])
        text_start_pos = len(self.input_ids)
        while True:
            next_token = next(self.llm.generate(self.input_ids[-1:], reset=False))
            self.input_ids.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                self.llm.eval([next_token])
                break
        if self.post_eos_input_ids:
            self.input_ids.extend(self.post_eos_input_ids)
            self.llm.eval(self.post_eos_input_ids)
        text = self.tokenizer.decode(self.input_ids[text_start_pos:], skip_special_tokens=True)
        text = text.strip()
        self.messages.append({
            "role": "assistant",
            "content": text
        })
        return text