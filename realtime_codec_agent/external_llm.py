from typing import Optional, Tuple, Dict, List
from .realtime_agent_resources import RealtimeAgentResources
from .realtime_agent_config import RealtimeAgentConfig

class ExternalLLMHandler:
    def __init__(
        self, 
        resources: RealtimeAgentResources,
        config: RealtimeAgentConfig,
    ):
        self.resources = resources
        self.config = config

        additional_instructions = f"\n\n## Instructions:\n{self.config.external_llm_instructions}" if self.config.external_llm_instructions else ""
        self.system_message = {
            "role": "system",
            "content": f"""You are a friendly assistant engaging in a spoken telephone conversation with a user.

## Response Format:
- Respond naturally, including backchannels (e.g. yeah, sure, mhm), fillers (e.g. uh, um, hmm) and laughter (e.g. [laughing], [laughs]).
- You can also choose to say nothing, in which case respond with [silence].
- If the user responds with a backchannel (e.g. yeah, sure, mhm) or with [silence], you may continue your previous response.{additional_instructions}"""
        }

        self.dummy_sys_msg, self.dummy_sys_msg_length, self.gen_prompt_input_ids, self.post_eos_input_ids = self._get_chat_template_stuff()
        self.sentence_boundary_tokens = set([self.resources.external_llm_tokenizer.encode(c, add_special_tokens=False)[0] for c in [".", "!", "?", "\n"]])
        self.start_audio_token_id = self.resources.tokenizer.convert_tokens_to_ids(self.config.start_audio_token)
        self.agent_identity_input_ids = self.resources.tokenizer.encode(f" {self.config.agent_identity}:", add_special_tokens=False)
        self.reset()

    def _get_chat_template_stuff(self) -> Tuple[Dict[str, str], int, List[int], List[int]]:
        dummy_messages = [
            {"role": "system", "content": "foo"},
            {"role": "user", "content": "bar"},
        ]
        sys_msg_input_ids = self.resources.external_llm_tokenizer.apply_chat_template(dummy_messages[:1])
        input_ids = self.resources.external_llm_tokenizer.apply_chat_template(dummy_messages)
        input_ids_with_gen_prompt = self.resources.external_llm_tokenizer.apply_chat_template(dummy_messages, add_generation_prompt=True)
        dummy_sys_msg = dummy_messages[0]
        dummy_sys_msg_length = len(sys_msg_input_ids)
        gen_prompt_input_ids = input_ids_with_gen_prompt[len(input_ids):]

        dummy_messages.append({"role": "assistant", "content": "baz"})
        prev_input_ids_length = len(input_ids)
        input_ids = self.resources.external_llm_tokenizer.apply_chat_template(dummy_messages)
        last_eos_pos = input_ids.index(self.resources.external_llm_tokenizer.eos_token_id, prev_input_ids_length)
        post_eos_input_ids = input_ids[last_eos_pos+1:]

        return dummy_sys_msg, dummy_sys_msg_length, gen_prompt_input_ids, post_eos_input_ids

    def reset(self) -> None:
        self.resources.external_llm.init_sampler_for_generate(
            top_p=self.config.external_llm_top_p,
            min_p=0.0,
            temp=1.0,
            seed=self.config.seed,
        )
        self.resources.external_llm.reset()
        self.messages = [self.system_message]
        if self.config.agent_opening_text:
            self.messages.extend([
                {
                    "role": "user",
                    "content": "[silence]",
                },
                {
                    "role": "assistant",
                    "content": self.config.agent_opening_text,
                },
            ],
        )
        self.ext_input_ids = self.resources.external_llm_tokenizer.apply_chat_template(self.messages)
        self.resources.external_llm.eval(self.ext_input_ids)

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
        new_input_ids = self.resources.external_llm_tokenizer.apply_chat_template([self.dummy_sys_msg] + new_messages)[self.dummy_sys_msg_length:]
        self.ext_input_ids.extend(new_input_ids)
        self.resources.external_llm.eval(new_input_ids)

    def generate_response(self) -> Optional[List[int]]:
        # Record state at start in case we need to undo (e.g. if response is suppressed)
        n_tokens_at_start = self.resources.llm.n_tokens
        n_ext_tokens_at_start = self.resources.external_llm.n_tokens
        num_messages_at_start = len(self.messages)
        num_ext_input_ids_at_start = len(self.ext_input_ids)
        # If last message is not from user, add a [silence] user message
        if self.messages[-1]["role"] != "user":
            self.messages.append({
                "role": "user",
                "content": "[silence]",
            })
            new_input_ids = self.resources.external_llm_tokenizer.apply_chat_template([self.dummy_sys_msg] + self.messages[-1:])[self.dummy_sys_msg_length:]
            self.ext_input_ids.extend(new_input_ids)
            self.resources.external_llm.eval(new_input_ids)
        # Generate
        self.ext_input_ids.extend(self.gen_prompt_input_ids)
        self.resources.external_llm.eval(self.gen_prompt_input_ids[:-1])
        sent_start_pos = text_start_pos = len(self.ext_input_ids)
        input_ids = self.agent_identity_input_ids.copy()
        self.resources.llm.eval(input_ids)
        while True:
            ext_next_token = next(self.resources.external_llm.generate(self.ext_input_ids[-1:], reset=False))
            self.ext_input_ids.append(ext_next_token)
            if ext_next_token == self.resources.external_llm_tokenizer.eos_token_id or ext_next_token in self.sentence_boundary_tokens:
                sent_text = self.resources.external_llm_tokenizer.decode(self.ext_input_ids[sent_start_pos:], skip_special_tokens=True)
                sent_text = sent_text.lower().replace(",", "").replace(".", "").strip()
                sent_input_ids = self.resources.tokenizer.encode(f" {sent_text}", add_special_tokens=False)
                next_token = next(self.resources.llm.generate(sent_input_ids, reset=False))
                input_ids.extend(sent_input_ids)
                sent_start_pos = len(self.ext_input_ids)
                if ext_next_token != self.resources.external_llm_tokenizer.eos_token_id and next_token == self.start_audio_token_id:
                    self.resources.external_llm.eval([ext_next_token])
                    ext_next_token = self.resources.external_llm_tokenizer.eos_token_id
                    self.ext_input_ids.append(ext_next_token)
            if ext_next_token == self.resources.external_llm_tokenizer.eos_token_id:
                self.resources.external_llm.eval([ext_next_token])
                text_end_pos = len(self.ext_input_ids)
                break
        if self.post_eos_input_ids:
            self.ext_input_ids.extend(self.post_eos_input_ids)
            self.resources.external_llm.eval(self.post_eos_input_ids)
        text = self.resources.external_llm_tokenizer.decode(self.ext_input_ids[text_start_pos:text_end_pos], skip_special_tokens=True)
        self.messages.append({
            "role": "assistant",
            "content": text
        })
        if text.startswith("[silen"):
            # Response is suppressed - undo everything and return None
            self.resources.llm.n_tokens = n_tokens_at_start
            self.resources.external_llm.n_tokens = n_ext_tokens_at_start
            self.messages = self.messages[:num_messages_at_start]
            self.ext_input_ids = self.ext_input_ids[:num_ext_input_ids_at_start]
            return None
        return input_ids