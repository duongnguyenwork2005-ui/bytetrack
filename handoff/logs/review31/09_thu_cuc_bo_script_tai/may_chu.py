import functools, http.server, os, re, sys


class H(http.server.SimpleHTTPRequestHandler):
    extensions_map = {".zip": "application/zip", ".md5": "text/plain", ".html": "text/html", "": "application/octet-stream"}

    def log_message(self, *a):
        pass

    def do_GET(self):
        p = self.translate_path(self.path)
        rng = self.headers.get("Range")
        if not os.path.isfile(p) or not p.endswith(".zip") or not os.path.exists(p + ".range"):
            return super().do_GET()
        # chi file co marker <ten>.range moi ho tro Range; marker <ten>.cat = so byte gui roi ngat ket noi (1 lan)
        data = open(p, "rb").read()
        L = len(data)
        bat_dau = 0
        if rng:
            m = re.fullmatch(r"bytes=(\d+)-", rng.strip())
            bat_dau = int(m.group(1))
            if bat_dau >= L:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{L}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {bat_dau}-{L - 1}/{L}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(L - bat_dau))
        self.end_headers()
        cat = p + ".cat"
        if os.path.exists(cat):
            k = int(open(cat).read())
            os.remove(cat)
            self.wfile.write(data[bat_dau:bat_dau + k])
            self.wfile.flush()
            self.connection.shutdown(2)
            self.close_connection = True
            return
        self.wfile.write(data[bat_dau:])


port, thu_muc = int(sys.argv[1]), sys.argv[2]
http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory=thu_muc)).serve_forever()
