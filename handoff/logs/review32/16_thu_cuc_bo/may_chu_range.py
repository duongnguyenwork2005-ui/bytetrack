# May chu gia lap ho tro Range dong "bytes=a-b" va mo "bytes=a-" (nhu CDN that), chi de kiem thu cuc bo.
import functools, http.server, os, re, sys


class H(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.translate_path(self.path)
        if not os.path.isfile(p):
            return super().do_GET()
        data = open(p, "rb").read()
        L = len(data)
        rng = self.headers.get("Range")
        if not rng:
            self.send_response(200)
            self.send_header("Content-Length", str(L))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)
            return
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng.strip())
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else L - 1
        b = min(b, L - 1)
        if a >= L:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{L}")
            self.end_headers()
            return
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{L}")
        self.send_header("Content-Length", str(b - a + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(data[a:b + 1])


port, thu_muc = int(sys.argv[1]), sys.argv[2]
http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory=thu_muc)).serve_forever()
