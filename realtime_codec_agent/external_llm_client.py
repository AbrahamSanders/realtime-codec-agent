from typing import List, Dict, Any, Optional
from openai import OpenAI
import threading

class ExternalLLMClient:
    @classmethod
    def get_models(cls, api_key: str, base_url: str) -> List[str]:
        try:
            client = OpenAI(
                api_key=api_key, 
                base_url=base_url,
            )
            available_models = client.models.list().data
            model_names = [model.id for model in available_models]
            return model_names
        except:
            return []

    def __init__(self, api_key: str, base_url: str, model: Optional[str] = None, agent_identity: str = "A", allow_laughter: bool = True):
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url,
        )
        is_openai = "openai.com" in self.client.base_url.host
        self.system_role = "developer" if is_openai else "system"
        self.assistant_prefill_supported = not is_openai
        if not model:
            available_models = ExternalLLMClient.get_models(api_key, base_url)
            if len(available_models) == 0:
                raise ValueError(f"No models found at {self.client.base_url}.")
            model = available_models[0]
        self.model = model
        self.agent_identity = agent_identity
        self.allow_laughter = allow_laughter
        self.cancelled_threads = set()
        self.prep_stream_thread = None
        self.stream = None
        self.stream_read_count = 0

    def get_messages(self, transcript: List[Dict[str, Any]], additional_instructions: str) -> List[Dict[str, Any]]:
        additional_instructions = f"\n\n## Instructions:\n{additional_instructions}" if additional_instructions else ""
        allow_laughter_instr = " and laughter (e.g. [laughing], [laughs] or &=laughing, &=laughs)" if self.allow_laughter else ""
        messages = [
            {
                "role": self.system_role,
                "content": f"""You are a friendly assistant engaging in a spoken telephone conversation with a user.

## Response Format:
- Respond naturally, including backchannels (e.g. yeah, sure, mhm) and fillers (e.g. uh, um, hmm){allow_laughter_instr}.
- You can also choose to say nothing, in which case respond with [silence].
- If the user responds with a backchannel (e.g. yeah, sure, mhm) or with [silence], you may continue your previous response.{additional_instructions}"""
            }
        ]
        for turn in transcript:
            if turn["speaker"] != self.agent_identity:
                if messages[-1]["role"] == "user":
                    messages[-1]["content"] += " " + turn["text"]
                else:               
                    messages.append({
                        "role": "user",
                        "content": turn["text"]
                    })
            else:
                if messages[-1]["role"] == self.system_role:
                    messages.append({
                        "role": "user",
                        "content": "[silence]"
                    })
                if messages[-1]["role"] == "assistant":
                    messages[-1]["content"] += " " + turn["text"]
                else:
                    messages.append({
                        "role": "assistant",
                        "content": turn["text"]
                    })
        # Ensure there is at least one non-system message, and that the last message is from the user if assistant prefill is not supported
        if len(messages) == 1 or (not self.assistant_prefill_supported and messages[-1]["role"] == "assistant"):
            messages.append({
                "role": "user",
                "content": "[silence]"
            })
        return messages

    def _prep_stream(self, messages: List[Dict[str, Any]], top_p: float, max_tokens: int) -> None:
        curr_thread = threading.current_thread()
        stream = None
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
            )
            if curr_thread in self.cancelled_threads:
                # if someone cancelled the stream before it was ready, dispose the connection because it won't be used
                stream.close()
                return
            self.stream = stream
            self.stream_read_count = 0
        except Exception as e:
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass
            raise e
        finally:
            if self.prep_stream_thread == curr_thread:
                self.prep_stream_thread = None
            self.cancelled_threads.discard(curr_thread)
            
    def prep_stream(self, transcript: List[Dict[str, Any]], additional_instructions: str, top_p: float = 0.9, max_tokens: int = 100) -> None:
        try:
            self.close_stream()
            messages = self.get_messages(transcript, additional_instructions)
            self.prep_stream_thread = threading.Thread(target=self._prep_stream, args=(messages, top_p, max_tokens), daemon=True)
            self.prep_stream_thread.start()
        except Exception as e:
            try:
                self.close_stream()
            except Exception:
                pass
            raise e

    def next_chunk(self) -> Optional[str]:
        if self.prep_stream_thread is not None:
            self.prep_stream_thread.join()
        if self.stream is None:
            return None
        while True:
            next_chunk = next(self.stream, None)
            if next_chunk is None:
                self.close_stream()
                return None
            chunk_text = next_chunk.choices[0].delta.content
            if not chunk_text:
                continue
            self.stream_read_count += 1
            return chunk_text
        
    def next_sentence(self) -> Optional[str]:
        sentence_chunks = []
        while True:
            chunk = self.next_chunk()
            if chunk is None:
                break
            sentence_chunks.append(chunk)
            if any(chunk.endswith(punct) for punct in [".", "!", "?", ":", ";"]):
                break
        sentence = "".join(sentence_chunks)
        sentence = sentence.replace("\n", " ").replace("[ ", "[").replace(" ]", "]").strip()
        return sentence if sentence else None        

    def close_stream(self, blocking: bool = False) -> None:
        if self.prep_stream_thread is not None:
            self.cancelled_threads.add(self.prep_stream_thread)
            self.prep_stream_thread = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        if blocking:
            for thread in list(self.cancelled_threads):
                thread.join()
