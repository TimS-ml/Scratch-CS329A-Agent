
import os
import re
import requests
from datetime import datetime
import json
from googleapiclient.discovery import build
from textblob import TextBlob
from pydantic import BaseModel
from typing import Dict, Any, List

from cs329a_hw3.api_manager import APIManager
from cs329a_hw3.utils import generate_together


class DeepResearchAgent:
    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager

    def generate(
        self,
        query: str,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        temperature: float = 0.7,
    ) -> str:
        """Single-call baseline: answer the query directly with one LM call (no tools)."""
        messages = [
            {"role": "system", "content": "You are a helpful research assistant. Answer clearly and concisely."},
            {"role": "user", "content": query},
        ]
        response_message = generate_together(model=model, messages=messages, temperature=temperature)
        return response_message.content if response_message else ""

    def research(self, query: str, max_searches: int = 4) -> Dict[str, Any]:
        """
        Conducts deep research on a given query.

        Args:
            query: Complex research question to investigate

        Returns:
            Dictionary containing:
            - report: str, synthesized findings with citations
            - sources: List[str], list of source URLs or references
        """
        # 1) Plan a handful of focused web searches that together cover the question.
        plan_prompt = (
            "You are a research planner. Propose up to "
            f"{max_searches} focused, diverse web-search queries that together will let you "
            "answer the research question thoroughly (cover sub-topics, key entities, dates). "
            "Wrap each query in <q>...</q> tags and output nothing else.\n\n"
            f"Research question: {query}"
        )
        plan_msg = generate_together(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": plan_prompt}],
            temperature=0.3,
        )
        plan_text = plan_msg.content if plan_msg else ""
        search_queries = [s.strip() for s in re.findall(r"<q>(.*?)</q>", plan_text, flags=re.DOTALL) if s.strip()]
        if not search_queries:
            search_queries = [query]
        search_queries = search_queries[:max_searches]

        # 2) Gather evidence from the web via the API manager's search tool.
        evidence_blocks: List[str] = []
        sources: List[str] = []
        for sq in search_queries:
            hits = self.api_manager.google_search(sq, num_results=3)
            for h in hits:
                if not isinstance(h, dict) or "error" in h:
                    continue
                link = h.get("link", "")
                if link:
                    sources.append(link)
                evidence_blocks.append(
                    f"Source: {link}\n"
                    f"Title: {h.get('title', '')}\n"
                    f"Snippet: {h.get('snippet', '')}\n"
                    f"Content: {h.get('content', '')[:1200]}"
                )
        evidence_text = "\n\n".join(evidence_blocks[:12]) if evidence_blocks else "(no evidence gathered)"

        # 3) Synthesize a structured, cited report.
        report_prompt = (
            "Write a well-structured research report of 4-5 paragraphs that answers the question "
            "below using ONLY the evidence provided. Requirements:\n"
            "  - Clear structure (intro, themed body paragraphs, brief conclusion).\n"
            "  - Cite sources inline by their URL where claims are made.\n"
            "  - Be temporally accurate (mention relevant dates/years).\n"
            "  - Do not invent facts beyond the evidence.\n\n"
            f"Research question: {query}\n\n"
            f"Evidence:\n{evidence_text}\n\n"
            "Report:"
        )
        report_msg = generate_together(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"role": "user", "content": report_prompt}],
            temperature=0.4,
            max_tokens=2000,
        )
        report = report_msg.content if report_msg else ""

        # De-duplicate sources, preserving order.
        seen = set()
        unique_sources = []
        for s in sources:
            if s and s not in seen:
                seen.add(s)
                unique_sources.append(s)

        return {"report": report, "sources": unique_sources}
