import asyncio
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from autojudge_base import (
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    LlmConfigProtocol,
    MeasureSpec,
    NuggetBanksProtocol,
    Qrels,
    Report,
    Request,
)
from minima_llm import MinimaLlmConfig, MinimaLlmRequest, MinimaLlmResponse, OpenAIMinimaLlm

KEYWORD_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("Relevance"),
))


class NegationJudge:

    def judge(
        self,
        rag_responses: Iterable[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigProtocol,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        qrels: Optional[Qrels] = None,
        **kwargs: Any,
    ) -> Leaderboard:
        topic_titles: Dict[str, str] = {t.request_id: t.problem_statement or "" for t in rag_topics}
        expected_topic_ids: List[str] = list(topic_titles.keys())

        # Collect all requests with metadata
        requests_info: List[Tuple[str, str, MinimaLlmRequest]] = []  # (run_id, topic_id, request)
        for i, response in enumerate(rag_responses):
            query = topic_titles.get(response.metadata.topic_id, "")
            full_response = response.get_report_text().lower() if response.responses else ""
            #print(full_response)
            requests_info.append((
                response.metadata.run_id,
                response.metadata.topic_id,
                MinimaLlmRequest(
                    request_id=f"q{i}",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a relevance evaluator that evaluates the relevance between a query and a response.\n"
                                "Score relevance between each query and response (0 to 3). Consider each query and response pair separately.\n"
                                "Return ONLY one integer that represents the relevance between each query and response pair.\n"
                                "The first integer is the relevance between the first query and the response. The second integer is the relevance between the second query and the response."
                                "Example output: 3 1"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Response: {full_response}\n\n"
                                f"Query 1: {query}\n\n"
                                f"Query 2: Not about, related to, or relevant to the following: {query}"
                            ),
                        }
                    ],
                    temperature=0.5,
                ),
            ))

        # Run all LLM requests in batch
        # Convert base config to full MinimaLlmConfig for backend features (batching, retry, etc.)
        full_config = MinimaLlmConfig.from_dict(llm_config.raw) if llm_config.raw else MinimaLlmConfig.from_env()
        backend = OpenAIMinimaLlm(full_config)
        llm_results = asyncio.run(backend.run_batched([req for _, _, req in requests_info]))
        print(llm_results)

        # Build leaderboard from responses
        builder = LeaderboardBuilder(KEYWORD_SPEC)
        for (run_id, topic_id, _), result in zip(requests_info, llm_results):
            scores = self._parse_relevance(result)
            rel_score = 0
            difference = abs(scores[0] - scores[1])
            if difference == 3:
                rel_score = scores[0]
            elif difference== 2:
                rel_score = scores[0] - 0.02
            elif difference == 1: 
                rel_score = scores[0] - 0.1
            else:
                rel_score = scores[0] - 0.15

            builder.add(run_id=run_id, topic_id=topic_id, values={"Relevance": rel_score})

        return builder.build(expected_topic_ids=expected_topic_ids, on_missing="fix_aggregate")

    def _parse_relevance(self, result: Any) -> List[int]:
        if not isinstance(result, MinimaLlmResponse):
            print(f"[KeywordJudge] LLM error: {result}")
            return [0, 0]

        text = result.text.strip().lower()
        numbers = re.findall(r"\b([0-3])\b", text)

        if len(numbers) >= 2:
            return [int(n) for n in numbers[:2]]

        return [0, 0]
        