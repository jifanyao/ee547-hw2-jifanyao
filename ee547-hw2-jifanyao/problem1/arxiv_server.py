import json
import os
import sys
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs


Sample_dir = os.path.join(os.path.dirname(__file__), "sample_data")
Papers_file = os.path.join(Sample_dir, "papers.json")
Stats_file = os.path.join(Sample_dir, "corpus_analysis.json")

try:
    with open(Papers_file, "r") as f:
        papers = json.load(f)
except FileNotFoundError:
    papers = []

try:
    with open(Stats_file, "r") as f:
        stats = json.load(f)
except FileNotFoundError:
    stats = {}


class ArxivHandler(BaseHTTPRequestHandler):

    def Print_request(self, method, path, status, extra=""):
   
        current_time = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print(f"{current_time} {method} {path} - {status} {extra}")

    def send_json(self, code, payload):
       
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path_parts = parsed.path.strip("/").split("/")

        try:
            # 1. GET /papers
            if parsed.path == "/papers":
                brief = []
                for p in papers:
                    brief.append({
                        "arxiv_id": p.get("arxiv_id"),
                        "title": p.get("title"),
                        "authors": p.get("authors", []),
                        "categories": p.get("categories", [])
                    })
                self.Print_request("GET", self.path, "200 OK", f"({len(brief)} results)")
                self.send_json(200, brief)
                return

            # 2. GET /papers/{arxiv_id} 
            if len(path_parts) == 2 and path_parts[0] == "papers":
                arxiv_id = path_parts[1]
                for p in papers:
                    if p.get("arxiv_id") == arxiv_id:
                        self.Print_request("GET", self.path, "200 OK")
                        self.send_json(200, p)
                        return
                self.Print_request("GET", self.path, "404 Not Found")
                self.send_json(404, {"error": "Paper not found"})
                return

            # 3. GET /search?q=... 
            if path_parts[0] == "search":
                query_params = parse_qs(parsed.query)
                if "q" not in query_params:
                    self.Print_request("GET", self.path, "400 Bad Request")
                    self.send_json(400, {"error": "Missing query parameter q"})
                    return
                query = query_params["q"][0].strip().lower()
                if not query:
                    self.Print_request("GET", self.path, "400 Bad Request")
                    self.send_json(400, {"error": "Empty query"})
                    return

                terms = query.split()
                results = []

                for p in papers:
                    title = (p.get("title", "")).lower()
                    abstract = (p.get("abstract", "")).lower()

                    term_counts_title = []
                    term_counts_abs = []
                    term_counts_total = []

                    for t in terms:
                        count_in_title = title.count(t)
                        count_in_abstract = abstract.count(t)
                        total_count = count_in_title + count_in_abstract

                        term_counts_title.append(count_in_title)
                        term_counts_abs.append(count_in_abstract)
                        term_counts_total.append(total_count)

                    if all(c > 0 for c in term_counts_total):
                        match_score = sum(term_counts_total)
                        matches_in = []
                        if any(c > 0 for c in term_counts_title):
                            matches_in.append("title")
                        if any(c > 0 for c in term_counts_abs):
                            matches_in.append("abstract")

                        results.append({
                            "arxiv_id": p.get("arxiv_id"),
                            "title": p.get("title"),
                            "match_score": match_score,
                            "matches_in": matches_in
                        })

                self.Print_request("GET", self.path, "200 OK", f"({len(results)} results)")
                self.send_json(200, {"query": query, "results": results})
                return

            # 4. GET /stats -
            if parsed.path == "/stats":
                self.Print_request("GET", self.path, "200 OK")
                self.send_json(200, stats)
                return

           
            self.Print_request("GET", self.path, "404 Not Found")
            self.send_json(404, {"error": "Invalid endpoint"})

        except Exception as e:
            self.Print_request("GET", self.path, "500 Server Error")
            self.send_json(500, {"error": str(e)})


def run(port=8080):
    server = ThreadingHTTPServer(("0.0.0.0", port), ArxivHandler)
    print(f"Server running on port {port}...")
    server.serve_forever()


if __name__ == "__main__":
    try:
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    except ValueError:
        print("Invalid port; must be integer")
        sys.exit(1)
    run(port)
