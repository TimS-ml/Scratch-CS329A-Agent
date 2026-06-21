import requests
import json
import re
from googleapiclient.discovery import build
from geopy.geocoders import Nominatim
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import polygon
import time as _time
import cs329a_hw3.wolfram_api as wolfram_api
import nest_asyncio
from cs329a_hw3.utils import generate_together
import cs329a_hw3._llm_cache as _llm_cache


def _cached_tool(key: dict, producer):
    """Cache *successful* external tool results so the agent is reproducible and
    tool outputs are persisted for later analysis. Errors are not cached so that
    transient failures can retry on the next run."""
    if not _llm_cache._enabled():
        return producer()
    k = _llm_cache.make_key(key)
    hit = _llm_cache._read(k)
    if hit is not None and "response" in hit:
        return hit["response"]
    res = producer()
    is_err = (isinstance(res, dict) and "error" in res) or (
        isinstance(res, list) and len(res) == 1 and isinstance(res[0], dict) and "error" in res[0]
    )
    if not is_err:
        _llm_cache._write(k, {"key": k, "tag": "hw3-tool", "request": key,
                              "response": res, "ts": _time.time(), "latency_s": 0.0})
    return res

# Pydantic models for structured LLM parsing
class GoogleSearchParams(BaseModel):
    search_query: str
    num_results: int = Field(default=2, description="The number of results to return from the Google search")

class StockDataParams(BaseModel):
    ticker: str = Field(description="The stock ticker symbol to get data for, e.g. 'AAPL'")
    date: str = Field(description="The date to get stock data for, in the format 'YYYY-MM-DD'")

class ComputeParams(BaseModel):
    wolfram_query: str = Field(description="The mathematical computation / operation to perform, e.g. 'integrate cos(x)/sqrt(x) from 0 to 1'")

class GetWeatherParams(BaseModel):
    location: str = Field(description="The location to get weather data for, in the format 'City, Country'")
    date: str = Field(description="The date to get weather data for, in the format 'YYYY-MM-DD'")
    hour: str = Field(default="12", description="The hour of the day to get weather data for, in the format 'HH' (24-hour format)")

class APIResponse(BaseModel):
    api_name: str = Field(description="The name of the API to use for the query. Must be one of: 'google_search', 'get_stock_data', 'compute', 'get_weather'")


def _extract_json(text: str) -> Optional[dict]:
    """Robustly pull a JSON object out of an LLM response (handles fences/prose)."""
    if not text:
        return None
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, flags=re.IGNORECASE)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


class APIManager:
    """A unified class to manage various API interactions, using an LLM to route queries."""
    
    def __init__(
        self, 
        google_api_key: str, 
        google_cx_id: str, 
        polygon_api_key: str, 
        wolfram_app_id: str, 
        router_model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    ):
        """Initializes the APIManager with the necessary API keys."""
        
        with open("weather_codes.json", "r") as f:
            self._weather_codes = json.load(f)

        self.api_functions = {
            "google_search": (self.google_search, GoogleSearchParams), 
            "get_stock_data": (self.get_stock_data, StockDataParams),
            "compute": (self.compute, ComputeParams),
            "get_weather": (self.get_weather, GetWeatherParams)
        }
        ################ CODE STARTS HERE ################
        self.google_api_key = google_api_key
        self.google_cx_id = google_cx_id
        self.polygon_api_key = polygon_api_key
        self.wolfram_app_id = wolfram_app_id
        self.router_model = router_model

        # Polygon stock client.
        self.polygon_client = polygon.RESTClient(polygon_api_key) if polygon_api_key else None
        # Wolfram Alpha client.
        self.wolfram_client = wolfram_api.Client(wolfram_app_id) if wolfram_app_id else None
        # Geocoder for the weather tool (no key required).
        self.geolocator = Nominatim(user_agent="cs329a_hw3_agent")
        # Google Custom Search is optional; we fall back to DuckDuckGo if unavailable.
        self.google_service = None
        if google_api_key and google_cx_id:
            try:
                self.google_service = build("customsearch", "v1", developerKey=google_api_key)
            except Exception:
                self.google_service = None
        ################ CODE ENDS HERE ################

    def _parse_query_params(
        self,
        query: str, 
        function_name: str,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Uses an LLM to parse parameters for a given function from a user query."""
        ################ CODE STARTS HERE ################
        model = model or self.router_model
        func, params_model = self.api_functions[function_name]
        schema = params_model.model_json_schema()
        system = (
            "You extract structured parameters for an API call from a user query.\n"
            f"API function `{function_name}`: {func.__doc__}\n\n"
            "Return ONLY a single JSON object that conforms to this JSON schema "
            "(no prose, no markdown, no code fences):\n"
            f"{json.dumps(schema)}\n\n"
            "Infer sensible values from the query. Any date must be 'YYYY-MM-DD'. "
            "Any hour must be two-digit 24-hour 'HH'."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
        response_message = generate_together(
            model=model, messages=messages, temperature=0,
            response_format={"type": "json_object"},
        )
        if response_message is None or not response_message.content:
            return None
        data = _extract_json(response_message.content)
        if not isinstance(data, dict):
            return None
        try:
            return params_model(**data).model_dump()
        except Exception:
            return None
        ################ CODE ENDS HERE ################

    def route_query(self, query: str, model: Optional[str] = None) -> Dict:
        """Determines the appropriate API for a query, parses parameters, and executes the API call."""
        ################ CODE STARTS HERE ################
        model = model or self.router_model
        api_names = list(self.api_functions.keys())
        descriptions = "\n".join(f"- {n}: {self.api_functions[n][0].__doc__}" for n in api_names)
        system = (
            "You are an API router. Choose exactly ONE API best suited to answer the user's query.\n"
            f"Available APIs:\n{descriptions}\n\n"
            'Return ONLY JSON of the form {"api_name": "<one of: '
            + ", ".join(api_names) + '>"} with no other text.'
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]
        response_message = generate_together(
            model=model, messages=messages, temperature=0,
            response_format={"type": "json_object"},
        )
        api_name = None
        if response_message is not None and response_message.content:
            data = _extract_json(response_message.content)
            if isinstance(data, dict):
                api_name = data.get("api_name")
        if api_name not in self.api_functions:
            api_name = "google_search"  # safe default

        func, _ = self.api_functions[api_name]
        params = self._parse_query_params(query, api_name, model=model)
        if params is None:
            return {"api_used": api_name, "params": None, "results": None,
                    "error": "Failed to parse parameters."}
        try:
            results = func(**params)
            return {"api_used": api_name, "params": params, "results": results, "error": None}
        except Exception as e:  # noqa: BLE001
            return {"api_used": api_name, "params": params, "results": None, "error": str(e)}
        ################ CODE ENDS HERE ################
        
    def _extract_webpage_content(self, url: str) -> str:
        """Helper to scrape and clean text content from a URL."""
        ################ CODE STARTS HERE ################
        try:
            resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form"]):
                tag.decompose()
            text = " ".join(soup.get_text(separator=" ").split())
            return text[:2000]
        except Exception:
            return ""
        ################ CODE ENDS HERE ################
        
        
    def google_search(self, search_query: str, num_results: int = 2) -> List[Dict]:
        """Performs a Google search for general knowledge questions, recent events, or information not covered by other tools."""
        ################ CODE STARTS HERE ################
        def _impl():
            try:
                results: List[Dict] = []
                if self.google_service is not None:
                    res = self.google_service.cse().list(
                        q=search_query, cx=self.google_cx_id, num=min(num_results, 10)
                    ).execute()
                    for it in res.get("items", [])[:num_results]:
                        link = it.get("link", "")
                        results.append({
                            "title": it.get("title", ""),
                            "link": link,
                            "snippet": it.get("snippet", ""),
                            "content": self._extract_webpage_content(link) if link else "",
                        })
                else:
                    # Fallback: keyless DuckDuckGo search.
                    from ddgs import DDGS
                    hits = list(DDGS().text(search_query, max_results=num_results))
                    for h in hits[:num_results]:
                        link = h.get("href") or h.get("link", "")
                        results.append({
                            "title": h.get("title", ""),
                            "link": link,
                            "snippet": h.get("body", "") or h.get("snippet", ""),
                            "content": self._extract_webpage_content(link) if link else "",
                        })
                return results if results else [{"error": "No search results found."}]
            except Exception as e:  # noqa: BLE001
                return [{"error": str(e)}]
        return _cached_tool({"tool": "google_search", "q": search_query, "n": num_results}, _impl)
        ################ CODE ENDS HERE ################

    def get_stock_data(self, ticker: str, date: str) -> Dict[str, Any]:
        """Retrieves historical stock data (open, high, low, close, volume) for a given ticker (e.g. 'AAPL') and YYYY-MM-DD date (e.g. '2025-03-05')."""
        ################ CODE STARTS HERE ################
        def _impl():
            try:
                if self.polygon_client is None:
                    return {"error": "Polygon client not configured."}
                aggs = list(self.polygon_client.list_aggs(ticker, 1, "day", date, date, limit=1))
                if not aggs:
                    return {"error": f"No stock data for {ticker} on {date} (market likely closed)."}
                a = aggs[0]
                return {
                    "ticker": ticker,
                    "date": date,
                    "open": a.open,
                    "high": a.high,
                    "low": a.low,
                    "close": a.close,
                    "volume": a.volume,
                }
            except Exception as e:  # noqa: BLE001
                return {"error": str(e)}
        return _cached_tool({"tool": "get_stock_data", "ticker": ticker, "date": date}, _impl)
        ################ CODE ENDS HERE ################
        
    def get_weather(self, location: str, date: str, hour: str = "12") -> Dict[str, Any]:
        """
        Fetches weather data (temperature, humidity, wind speed) for a specific location (e.g. 'Palo Alto, CA') and YYYY-MM-DD date (e.g. '2025-10-22') and HH hour (e.g. '15' for 3PM).
        """
        ################ CODE STARTS HERE ################
        def _impl():
            try:
                loc = self.geolocator.geocode(location)
                if loc is None:
                    return {"error": f"Could not geocode location: {location}"}
                lat, lon = loc.latitude, loc.longitude

                hourly_vars = "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
                common = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": date,
                    "end_date": date,
                    "hourly": hourly_vars,
                    "timezone": "auto",
                    "temperature_unit": "fahrenheit",
                    "wind_speed_unit": "mph",
                }
                # Historical archive first; fall back to the forecast endpoint for very
                # recent / future dates that the archive does not cover yet.
                hourly = {}
                for base in ("https://archive-api.open-meteo.com/v1/archive",
                             "https://api.open-meteo.com/v1/forecast"):
                    try:
                        r = requests.get(base, params=common, timeout=20)
                        j = r.json()
                        hourly = j.get("hourly", {}) or {}
                        if hourly.get("time"):
                            break
                    except Exception:
                        continue

                times = hourly.get("time", [])
                if not times:
                    return {"error": f"No weather data for {location} on {date}."}

                target = f"{date}T{int(hour):02d}:00"
                idx = times.index(target) if target in times else min(
                    range(len(times)),
                    key=lambda j: abs(int(times[j][11:13]) - int(hour)),
                )
                code = hourly.get("weather_code", [None] * len(times))[idx]
                return {
                    "location": location,
                    "date": date,
                    "hour": f"{int(hour):02d}:00",
                    "temperature": hourly.get("temperature_2m", [None] * len(times))[idx],
                    "weather": self._weather_codes.get(str(code), f"code {code}"),
                    "humidity": hourly.get("relative_humidity_2m", [None] * len(times))[idx],
                    "wind_speed": hourly.get("wind_speed_10m", [None] * len(times))[idx],
                }
            except Exception as e:  # noqa: BLE001
                return {"error": str(e)}
        return _cached_tool(
            {"tool": "get_weather", "location": location, "date": date, "hour": hour}, _impl
        )
        ################ CODE ENDS HERE ################

    def compute(self, wolfram_query: str) -> Dict[str, str]:
        """Performs a mathematical computation or operation using Wolfram Alpha."""
        def _impl():
            try:
                nest_asyncio.apply()
                ################ CODE STARTS HERE ################
                if self.wolfram_client is None:
                    return {"error": "Wolfram client not configured."}
                res = self.wolfram_client.query(wolfram_query)

                primary = None
                try:
                    primary = next(res.results).text
                except Exception:
                    primary = None

                details: Dict[str, str] = {}
                for pod in (res.pods or []):
                    try:
                        details[pod.title] = pod.text
                    except Exception:
                        continue

                out: Dict[str, Any] = {"query": wolfram_query}
                if primary:
                    out["result"] = primary
                if details:
                    out["details"] = details
                if not primary and not details:
                    return {"error": "Wolfram Alpha returned no interpretable result."}
                return out
                ################ CODE ENDS HERE ################
            except Exception as e:  # noqa: BLE001
                ################ CODE ENDS HERE ################
                return {"error": str(e)}
        return _cached_tool({"tool": "compute", "wolfram_query": wolfram_query}, _impl)
