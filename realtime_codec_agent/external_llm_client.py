from typing import List, Dict, Any, Optional
from collections import deque
from openai import OpenAI
import re
import threading

PENDING_STATUS = "{PENDING}"

class ExternalLLMClient:
    def __init__(self, api_key: str, base_url: str, model: str, agent_identity: str):
        self.sentence_split_regex = re.compile("(?<=[.!?:]) ")
        self.pending_status = PENDING_STATUS
        self.client = OpenAI(
            api_key=api_key, 
            base_url=base_url,
        )
        self.model = model
        self.agent_identity = agent_identity
        self.prep_stream_threads = {}
        self.prep_stream_lock = threading.Lock()
        self.stream_resp = None
        self.stream = None
        self.stream_access_count = None

    def _get_messages(self, transcript: List[Dict[str, Any]], additional_instructions: str) -> List[Dict[str, Any]]:
        is_openai = "openai.com" in self.client.base_url.host
        additional_instructions = f"\n\n## Instructions:\n{additional_instructions}" if additional_instructions else ""
        messages = [
            {
                "role": "developer" if is_openai else "system",
                "content": f"""You are a friendly assistant engaging in a spoken telephone conversation with a user.

## Response Format:
- Respond naturally, including backchannels (e.g. yeah, sure, mhm), fillers (e.g. uh, um, hmm) and laughter (e.g. [laughing], [laughs]).
- You can also choose to say nothing, in which case respond with [silence].
- If the user responds with a backchannel (e.g. yeah, sure, mhm) or with [silence], you may continue your previous response.{additional_instructions}"""
            }
        ]
        for turn in transcript:
            if turn["speaker"] != self.agent_identity:
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
    
    def _iter_sentences(self):
        response_buffer = ""
        response_sentences = deque()
        for chunk in self.stream_resp:
            response_buffer += chunk.choices[0].delta.content or ""
            split_result = self.sentence_split_regex.split(response_buffer)
            if len(split_result) > 1:
                response_sentences.extend(split_result[:-1])
                response_buffer = split_result[-1]
            while len(response_sentences) > 0:
                yield response_sentences.popleft().replace("\n", " ").strip()
        if response_buffer:
            yield response_buffer.replace("\n", " ").strip()

    def _prep_stream(self, messages: List[Dict[str, Any]], top_p: float, max_tokens: int) -> None:
        curr_thread = threading.current_thread()
        stream_resp = None
        try:
            stream_resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=True,
            )
            if not self.prep_stream_threads[curr_thread]:
                # if someone cancelled the stream before it was ready, dispose the connection because it won't be used
                stream_resp.close()
                return
            with self.prep_stream_lock:
                self.stream_resp = stream_resp
                self.stream = self._iter_sentences()
                self.stream_access_count = 0
        except Exception as e:
            try:
                if stream_resp is not None:
                    stream_resp.close()
            except Exception:
                pass
            raise e
        finally:
            with self.prep_stream_lock:
                self.prep_stream_threads.pop(curr_thread, None)
            
    def prep_stream(self, transcript: List[Dict[str, Any]], additional_instructions: str, top_p: float = 0.9, max_tokens: int = 100) -> None:
        try:
            self.close_stream()
            messages = self._get_messages(transcript, additional_instructions)
            prep_stream_thread = threading.Thread(target=self._prep_stream, args=(messages, top_p, max_tokens), daemon=True)
            self.prep_stream_threads[prep_stream_thread] = True
            prep_stream_thread.start()
        except Exception as e:
            try:
                self.close_stream()
            except Exception:
                pass
            raise e
        
    def next_sentence(self) -> Optional[str]:
        if self.stream is None:
            with self.prep_stream_lock:
                if any(self.prep_stream_threads.values()):
                    return self.pending_status
            return None
        try:
            sentence = next(self.stream, None)
            self.stream_access_count += 1
            if sentence is None:
                self.close_stream()
            return sentence
        except Exception as e:
            try:
                self.close_stream()
            except Exception:
                pass
            raise e

    def close_stream(self, blocking: bool = False) -> None:
        with self.prep_stream_lock:
            for prep_stream_thread in self.prep_stream_threads:
                self.prep_stream_threads[prep_stream_thread] = False
            if self.stream_resp is not None:
                self.stream_resp.close()
                self.stream_resp = None
                self.stream = None
                self.stream_access_count = None
        if blocking:
            with self.prep_stream_lock:
                join_threads = list(self.prep_stream_threads)
            for thread in join_threads:
                thread.join()
